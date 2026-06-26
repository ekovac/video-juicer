# BD-J navigation probes (experimental, parked)

Three small libbluray/C tools for the case where a Blu-ray's episode **order**
lives only in BD-J (Blu-ray Java) navigation code — invisible to the static
`.mpls` parsing in `discs.py`. Hit on **The Magicians** (tmdb 64432): every disc
is BD-J authored with playlist obfuscation, no play-all playlist, and
same-runtime episodes, so metadata can't verify order and `order_by_playall`
correctly finds nothing.

**Status: parked, partial.** Headless BD-J *boot + navigation* works and is the
hard part. Full play-all *order extraction* does not work yet. Kept in our back
pocket for otherwise-hopeless series. For The Magicians we fell back to OCR /
positional order instead.

## Build

```sh
gcc -O2 -o bdj_probe    bdj_probe.c    $(pkg-config --cflags --libs libbluray)
gcc -O2 -o bdj_navigate bdj_navigate.c $(pkg-config --cflags --libs libbluray)
gcc -O2 -o bdj_titlemap bdj_titlemap.c $(pkg-config --cflags --libs libbluray)
```

## The headless-BD-J recipe (the reusable win)

libbluray *can* run BD-J headless, but two things must be right or it dies on
init:

1. **Use JDK 21, NOT the system default.** Java 24+ **removed `SecurityManager`**
   (`System.implSetSecurityManager` is gone), which BD-J hard-depends on — boot
   aborts with "Failed initializing SecurityManager". JDK 21 still has it
   (libbluray 1.4.0 has a "Java >= 18 setSecurityManager workaround").
2. **Force AWT headless.** No X display → the X11 toolkit (`libawt_xawt.so`)
   can't load (and some JDKs don't even ship it). `-Djava.awt.headless=true`
   makes it use `libawt_headless.so`.

```sh
export JAVA_HOME=/usr/lib/jvm/java-21-temurin-jdk     # any JDK <= 21
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=true -Xlog:disable"   # -Xlog: silences JNI spam
./bdj_navigate "<disc-dir>" [wall_secs] [start_title]
```

libbluray picks the JVM via `JAVA_HOME`; the JVM reads `JAVA_TOOL_OPTIONS` at
startup, so that's how we inject headless mode into libbluray's embedded VM.

## The tools

- **`bdj_probe`** — enumerates `index.bdmv` title objects (HDMV vs BD-J, First
  Play, Top Menu) + the playlist→duration/clip-count table. No JVM needed; pure
  structural read. Use it to confirm "is this BD-J?" and spot the decoys
  (looping playlists: long duration but 1-2 unique clips spliced 76-252x).
- **`bdj_navigate`** — boots First Play (or `bd_play_title(start_title)`),
  pumps the libbluray event loop, logs TITLE/PLAYLIST/MENU/etc., and when an
  *episode-length, low-clip-count* playlist starts it seeks to near the end to
  try to ride the app's auto-advance through the season. Drives the menu with
  ENTER/DOWN/RIGHT on IDLE/STILL.
- **`bdj_titlemap`** — jumps to each index title via `bd_play_title()` and
  records the first episode playlist it lands on. A clean title→episode mapping
  would give order with no blind menu navigation.

## What we learned on The Magicians S1D1

- BD-J confirmed: 79/81 title objects BD-J; First Play + Top Menu both BD-J.
- Episodes are playlists `00800`–`00803` (single-clip, 42-52 min). The long
  `00013`/`00020` playlists are menu-background **decoys** (one clip looped
  76-252x) — distinguished from episodes by `clip_count <= 8`.
- **Anchor recovered:** feature `title 1` plays `00800` = **E01**. So the lowest
  episode playlist number is episode 1 (consistent with positional order).

## Why full order extraction stalls (the open problem)

- `title 1` plays **only E01**, then returns to the menu — it's "Play Episode 1",
  not a chained "Play All". No auto-advance to E02 to ride.
- The per-episode BD-J titles (`id_ref 2`) won't play standalone: they're gated
  behind menu **register state** (PSR/GPR) the menu normally sets before
  launching the player xlet.
- Triggering the real "Play All" means either (a) driving the menu *blind*
  (fragile; and our max-speed `bd_read_ext` fights the menu's real-time
  animation/state), or (b) reverse-engineering which GPR the player xlet reads
  for "which episode / play-all" and setting it via `bd_play_title` + register
  writes. Both are disc-authoring-specific.

## If we pick this back up

- Register the ARGB overlay buffer and actually capture menu graphics, so menu
  navigation isn't blind (read button focus / "Play All" position).
- Throttle reads toward real-time while a menu is up so the BD-J app behaves.
- Probe/script GPR writes to unlock the per-episode titles, then map
  title→episode directly (the cleanest path if the register is discoverable).
- Generalize the JDK-selection (auto-pick a `<=21` JVM) into the harness.
