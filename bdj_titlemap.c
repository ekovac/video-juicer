/* bdj_titlemap.c — for each index title object, jump to it via bd_play_title()
 * and record the first *episode* playlist it plays. A clean title->episode
 * mapping means the title order is the broadcast order — recoverable without
 * any blind menu navigation.
 *
 * Build: gcc -O2 -o bdj_titlemap bdj_titlemap.c $(pkg-config --cflags --libs libbluray)
 * Run:   JAVA_HOME=/usr/lib/jvm/java-21-temurin-jdk \
 *        JAVA_TOOL_OPTIONS="-Djava.awt.headless=true -Xlog:disable" \
 *        ./bdj_titlemap "<disc-dir>"
 */
#include <libbluray/bluray.h>
#include <libbluray/keys.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#define BUFLEN      (6144 * 64)
#define MIN_EP_SECS 600.0
#define PER_TITLE_S 8

static double secs(uint64_t t) { return (double)t / 90000.0; }

static uint32_t pl_id[1024]; static uint64_t pl_dur[1024];
static uint32_t pl_clips[1024]; static int pl_n = 0;
static int is_episode(uint32_t pl) {
    for (int i = 0; i < pl_n; i++)
        if (pl_id[i] == pl) return secs(pl_dur[i]) > MIN_EP_SECS && pl_clips[i] <= 8;
    return 0;
}
static double dur_min(uint32_t pl) {
    for (int i = 0; i < pl_n; i++) if (pl_id[i] == pl) return secs(pl_dur[i]) / 60.0;
    return 0;
}

static void overlay_cb(void *h, const struct bd_overlay_s *const o) { (void)h; (void)o; }
static void argb_cb(void *h, const struct bd_argb_overlay_s *const o) { (void)h; (void)o; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <disc-dir>\n", argv[0]); return 2; }
    BLURAY *bd = bd_open(argv[1], NULL);
    if (!bd) { fprintf(stderr, "bd_open failed\n"); return 1; }

    const BLURAY_DISC_INFO *di = bd_get_disc_info(bd);
    uint32_t num_titles = di ? di->num_titles : 0;

    uint32_t nt = bd_get_titles(bd, TITLES_ALL, 0);
    for (uint32_t i = 0; i < nt && pl_n < 1024; i++) {
        BLURAY_TITLE_INFO *ti = bd_get_title_info(bd, i, 0);
        if (!ti) continue;
        pl_id[pl_n] = ti->playlist; pl_dur[pl_n] = ti->duration;
        pl_clips[pl_n] = ti->clip_count; pl_n++;
        bd_free_title_info(ti);
    }

    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_REGION_CODE, 1);
    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_PARENTAL, 99);
    bd_set_player_setting(bd, BLURAY_PLAYER_SETTING_PLAYER_PROFILE, 0x070200);
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_AUDIO_LANG, "eng");
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_MENU_LANG, "eng");
    bd_set_player_setting_str(bd, BLURAY_PLAYER_SETTING_COUNTRY_CODE, "US");
    bd_register_overlay_proc(bd, NULL, overlay_cb);
    bd_register_argb_overlay_proc(bd, NULL, argb_cb, NULL);

    if (bd_play(bd) <= 0) { fprintf(stderr, "bd_play failed\n"); bd_close(bd); return 1; }

    uint8_t *buf = malloc(BUFLEN);
    BD_EVENT ev;
    printf("=== TITLE -> first episode playlist ===\n");
    for (uint32_t t = 1; t <= num_titles; t++) {
        const BLURAY_TITLE *to = di->titles ? di->titles[t] : NULL;
        if (to && to->bdj == 0) continue;   /* HDMV trivia titles: skip noise */
        if (bd_play_title(bd, t) <= 0) continue;

        time_t start = time(NULL);
        uint32_t found = 0; uint32_t seen_pl = 0xffffffff;
        while (time(NULL) - start < PER_TITLE_S) {
            int n = bd_read_ext(bd, buf, BUFLEN, &ev);
            if (n < 0) break;
            if (ev.event == BD_EVENT_PLAYLIST) {
                seen_pl = ev.param;
                if (is_episode(seen_pl)) { found = seen_pl; break; }
            }
            /* nudge through any logo/still gate */
            if (ev.event == BD_EVENT_IDLE || ev.event == BD_EVENT_STILL)
                bd_user_input(bd, -1, BD_VK_ENTER);
            if (n == 0 && ev.event == BD_EVENT_NONE) usleep(4000);
        }
        if (found)
            printf("  title %2u (id_ref %u): EPISODE %05u.mpls (%.1f min)\n",
                   t, to ? to->id_ref : 0, found, dur_min(found));
        else if (seen_pl != 0xffffffff)
            printf("  title %2u (id_ref %u): playlist %05u.mpls (%.1f min, not episode)\n",
                   t, to ? to->id_ref : 0, seen_pl, dur_min(seen_pl));
    }

    free(buf);
    bd_close(bd);
    return 0;
}
