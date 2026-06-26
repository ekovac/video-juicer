/* bdj_probe.c — enumerate a Blu-ray's index.bdmv title objects + playlists
 * via libbluray, to reveal a play-all that static .mpls parsing can't see.
 *
 * Build: gcc -O2 -o bdj_probe bdj_probe.c $(pkg-config --cflags --libs libbluray)
 * Run:   ./bdj_probe "<path-to-BDMV-parent-dir>"
 */
#include <libbluray/bluray.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static double secs(uint64_t ticks) { return (double)ticks / 90000.0; }

static void print_title_obj(const char *tag, const BLURAY_TITLE *t) {
    if (!t) { printf("  %-10s (none)\n", tag); return; }
    printf("  %-10s id_ref=%-5u %-5s interactive=%u accessible=%u hidden=%u%s%s\n",
           tag, t->id_ref, t->bdj ? "BD-J" : "HDMV",
           t->interactive, t->accessible, t->hidden,
           t->name && t->name[0] ? " name=" : "", t->name ? t->name : "");
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <disc-dir>\n", argv[0]); return 2; }
    const char *path = argv[1];

    BLURAY *bd = bd_open(path, NULL);
    if (!bd) { fprintf(stderr, "bd_open failed for %s\n", path); return 1; }

    const BLURAY_DISC_INFO *di = bd_get_disc_info(bd);
    if (!di) { fprintf(stderr, "bd_get_disc_info returned NULL\n"); bd_close(bd); return 1; }

    printf("=== DISC INFO: %s ===\n", path);
    printf("  bluray_detected=%u  first_play_supported=%u  top_menu_supported=%u\n",
           di->bluray_detected, di->first_play_supported, di->top_menu_supported);
    printf("  bdj_detected=%u  bdj_supported=%u  libjvm_detected=%u  bdj_handled=%u\n",
           di->bdj_detected, di->bdj_supported, di->libjvm_detected, di->bdj_handled);
    printf("  num_hdmv_titles=%u  num_bdj_titles=%u  num_unsupported=%u\n",
           di->num_hdmv_titles, di->num_bdj_titles, di->num_unsupported_titles);

    printf("\n=== INDEX TITLE OBJECTS (from index.bdmv) : %u ===\n", di->num_titles);
    print_title_obj("FirstPlay", di->first_play);
    print_title_obj("TopMenu",   di->top_menu);
    if (di->titles) {
        for (uint32_t i = 1; i <= di->num_titles; i++) {
            const BLURAY_TITLE *t = di->titles[i];
            if (!t) continue;
            char tag[16]; snprintf(tag, sizeof tag, "Title#%u", i);
            print_title_obj(tag, t);
        }
    }

    /* Player-visible titles -> playlist + duration. TITLES_ALL = every playlist
     * the nav layer can reach; a play-all would surface here as a long title. */
    uint32_t n = bd_get_titles(bd, TITLES_ALL, 0);
    printf("\n=== PLAYER TITLES (bd_get_titles ALL) : %u ===\n", n);
    printf("  %-4s %-9s %9s %6s %6s\n", "idx", "playlist", "dur(min)", "clips", "chaps");
    for (uint32_t i = 0; i < n; i++) {
        BLURAY_TITLE_INFO *ti = bd_get_title_info(bd, i, 0);
        if (!ti) continue;
        printf("  %-4u %05u.mpls %9.1f %6u %6u\n",
               ti->idx, ti->playlist, secs(ti->duration) / 60.0,
               ti->clip_count, ti->chapter_count);
        bd_free_title_info(ti);
    }

    bd_close(bd);
    return 0;
}
