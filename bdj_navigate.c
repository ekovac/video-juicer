/* bdj_navigate.c — drive a Blu-ray's BD-J/HDMV navigation headlessly via
 * libbluray and log the playlist/title transitions, to recover the play-all
 * (== broadcast) order that lives only in BD-J code.
 *
 * Strategy: boot First Play, kick through logos/menus (send ENTER on
 * still/idle), and when an *episode-length* playlist (> MIN_EP_SECS) starts,
 * seek to near its end so the BD-J app auto-advances to the next episode. The
 * order episode playlists are visited is the broadcast order. Everything is
 * capped (wall clock, events, idle-kicks) so it can't hang.
 *
 * Build: gcc -O2 -o bdj_navigate bdj_navigate.c $(pkg-config --cflags --libs libbluray)
 * Run:   ./bdj_navigate "<disc-dir>" [wall_secs]
 */
#include <libbluray/bluray.h>
#include <libbluray/keys.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BUFLEN      (6144 * 64)
#define MIN_EP_SECS 600.0          /* >10 min playlist == an episode */
#define MAX_EVENTS  20000
#define MAX_KICKS   60
#define STALL_READS 150            /* consecutive dead reads before a kick */

static double secs(uint64_t t) { return (double)t / 90000.0; }

static const char *evname(uint32_t e) {
    switch (e) {
    case BD_EVENT_TITLE: return "TITLE";
    case BD_EVENT_PLAYLIST: return "PLAYLIST";
    case BD_EVENT_PLAYITEM: return "PLAYITEM";
    case BD_EVENT_CHAPTER: return "CHAPTER";
    case BD_EVENT_END_OF_TITLE: return "END_OF_TITLE";
    case BD_EVENT_STILL: return "STILL";
    case BD_EVENT_STILL_TIME: return "STILL_TIME";
    case BD_EVENT_IDLE: return "IDLE";
    case BD_EVENT_POPUP: return "POPUP";
    case BD_EVENT_MENU: return "MENU";
    case BD_EVENT_PLAYLIST_STOP: return "PLAYLIST_STOP";
    case BD_EVENT_ERROR: return "ERROR";
    case BD_EVENT_ENCRYPTED: return "ENCRYPTED";
    case BD_EVENT_READ_ERROR: return "READ_ERROR";
    default: return NULL;
    }
}

/* playlist -> duration(ticks)+clip_count, from a pre-scan, so we can seek to
 * end and tell a real episode from a looping menu-background decoy. */
static uint32_t pl_id[1024];
static uint64_t pl_dur[1024];
static uint32_t pl_clips[1024];
static int pl_n = 0;
static uint64_t dur_of(uint32_t pl) {
    for (int i = 0; i < pl_n; i++) if (pl_id[i] == pl) return pl_dur[i];
    return 0;
}
/* An episode: long, but with few clips. Decoys/menu loops are long too but
 * splice one or two clips dozens-to-hundreds of times (clip_count >> 8). */
static int is_episode(uint32_t pl) {
    for (int i = 0; i < pl_n; i++)
        if (pl_id[i] == pl)
            return secs(pl_dur[i]) > MIN_EP_SECS && pl_clips[i] <= 8;
    return 0;
}

/* episode playlists visited, in order (dedup consecutive + repeats) */
static uint32_t order[256]; static int order_n = 0;
static int already(uint32_t pl) {
    for (int i = 0; i < order_n; i++) if (order[i] == pl) return 1;
    return 0;
}

static void overlay_cb(void *h, const struct bd_overlay_s *const o) { (void)h; (void)o; }
static void argb_cb(void *h, const struct bd_argb_overlay_s *const o) { (void)h; (void)o; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <disc-dir> [wall_secs]\n", argv[0]); return 2; }
    const char *path = argv[1];
    int wall = argc > 2 ? atoi(argv[2]) : 120;

    BLURAY *bd = bd_open(path, NULL);
    if (!bd) { fprintf(stderr, "bd_open failed\n"); return 1; }

    /* pre-scan playlists -> durations */
    uint32_t nt = bd_get_titles(bd, TITLES_ALL, 0);
    for (uint32_t i = 0; i < nt && pl_n < 1024; i++) {
        BLURAY_TITLE_INFO *ti = bd_get_title_info(bd, i, 0);
        if (!ti) continue;
        pl_id[pl_n] = ti->playlist; pl_dur[pl_n] = ti->duration;
        pl_clips[pl_n] = ti->clip_count; pl_n++;
        bd_free_title_info(ti);
    }

    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_REGION_CODE, 1);   /* A */
    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_PARENTAL, 99);
    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_PLAYER_PROFILE, 0x070200);
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_AUDIO_LANG, "eng");
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_MENU_LANG, "eng");
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_PG_LANG, "eng");
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_COUNTRY_CODE, "US");
    bd_register_overlay_proc(bd, NULL, overlay_cb);
    bd_register_argb_overlay_proc(bd, NULL, argb_cb, NULL);

    if (bd_play(bd) <= 0) { fprintf(stderr, "bd_play failed\n"); bd_close(bd); return 1; }
    /* start the feature title directly (title 1 == the play-all entry); the
     * seek-to-end logic below then rides its auto-advance through the season */
    int start_title = argc > 3 ? atoi(argv[3]) : 1;
    bd_play_title(bd, start_title);

    uint8_t *buf = malloc(BUFLEN);
    BD_EVENT ev;
    long events = 0, kicks = 0;
    uint32_t cur_pl = 0xffffffff, last_log = 0xfffffffe;
    time_t start = time(NULL), last_kick = 0;
    int waiting = 0;   /* menu/idle/still: app wants user input */

    while (events < MAX_EVENTS) {
        time_t now = time(NULL);
        if (now - start > wall) { printf("# wall-clock cap (%ds)\n", wall); break; }
        if (order_n >= 16) { printf("# captured enough episodes\n"); break; }

        int n = bd_read_ext(bd, buf, BUFLEN, &ev);
        if (n < 0) { printf("# read error\n"); break; }

        if (ev.event != BD_EVENT_NONE) {
            const char *nm = evname(ev.event);
            /* collapse the IDLE/STILL floods; log everything else */
            int quiet = (ev.event == BD_EVENT_IDLE || ev.event == BD_EVENT_STILL ||
                         ev.event == BD_EVENT_STILL_TIME);
            if (nm && !(quiet && ev.event == last_log)) {
                printf("[%3lds] %-13s %u\n", now - start, nm, ev.param);
                events++;
            }
            last_log = ev.event;

            switch (ev.event) {
            case BD_EVENT_IDLE: case BD_EVENT_STILL: case BD_EVENT_MENU:
            case BD_EVENT_POPUP: case BD_EVENT_STILL_TIME:
                waiting = 1; break;
            case BD_EVENT_PLAYLIST:
                cur_pl = ev.param; waiting = 0;
                if (is_episode(cur_pl)) {
                    if (!already(cur_pl) && order_n < 256) order[order_n++] = cur_pl;
                    uint64_t dt = dur_of(cur_pl);
                    bd_seek_time(bd, dt - 12 * 90000);  /* -> near end, auto-advance */
                    printf("        -> EPISODE %05u.mpls (%.1f min); seek->end\n",
                           cur_pl, secs(dt) / 60.0);
                }
                break;
            case BD_EVENT_PLAYITEM: case BD_EVENT_CHAPTER: case BD_EVENT_TITLE:
                waiting = 0; break;
            case BD_EVENT_ERROR: case BD_EVENT_ENCRYPTED:
                printf("# fatal event\n"); goto done;
            }
            continue;
        }

        if (n > 0) { waiting = 0; continue; }   /* streaming */

        /* no data + no event: if the app is waiting, drive the menu */
        if (waiting && now - last_kick >= 1) {
            last_kick = now;
            if (++kicks > MAX_KICKS) { printf("# kick cap\n"); break; }
            /* try to activate the default ("Play"/"Play All") button; vary the
             * input so a non-default layout still gets exercised */
            uint32_t key = (kicks % 3 == 0) ? BD_VK_DOWN :
                           (kicks % 3 == 1) ? BD_VK_ENTER : BD_VK_RIGHT;
            bd_user_input(bd, -1, key);
            printf("        .. kick %ld: key=%s\n", kicks,
                   key == BD_VK_ENTER ? "ENTER" : key == BD_VK_DOWN ? "DOWN" : "RIGHT");
        }
        usleep(5000);
    }
done:;

    printf("\n=== EPISODE PLAYLIST ORDER (as navigated) ===\n");
    for (int i = 0; i < order_n; i++)
        printf("  %2d. %05u.mpls (%.1f min)\n", i + 1, order[i], secs(dur_of(order[i])) / 60.0);
    if (!order_n) printf("  (none captured)\n");

    free(buf);
    bd_close(bd);
    return 0;
}
