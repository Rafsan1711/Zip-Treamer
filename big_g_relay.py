# ba_meta require api 9
# -----------------------------------------------------------------------------
#
#                            Big G Relay Race
#
#   A minigame for BombSquad where teams race in a relay format on the
#   "Big G" map. Player spawn points are not predefined but are calculated
#   mathematically along the track's path to ensure even distribution,
#   regardless of the number of players per team.
#
#   Features:
#   - Dynamic Spawn Points: Uses vector math and linear interpolation to place
#     players accurately on the track surface.
#   - True Relay Mechanics: Only one player per team is active at a time.
#     The "baton" is passed by physical proximity.
#   - Scalable Gameplay: The number of laps and checkpoints automatically
#     adjusts to the number of players.
#   - Team-Based and Co-op: Works for both multi-team competition and
#     single-team cooperative play.
#
#   Author: Rafsan
#   Technical Implementation: Gemini
#   Version: 1.1.0
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, List, Tuple, Optional, Sequence

import babase
import bascenev1 as bs
import bauiv1 as bui

if TYPE_CHECKING:
    from typing import Any

# -----------------------------------------------------------------------------
#                      METADATA FOR THE PLUGIN MANAGER
# -----------------------------------------------------------------------------
plugman = {
    "plugin_name": "big_g_relay_rafsan",
    "description": "A mathematical relay race on the Big G map.",
    "authors": [{"name": "Rafsan"}],
    "version": "1.1.0",
}


# -----------------------------------------------------------------------------
#                     3D VECTOR MATH HELPER CLASS
# -----------------------------------------------------------------------------
class Vec3:
    """A minimalist helper class for 3D vector operations."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Vec3:
        l = self.length()
        if l == 0: return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

# -----------------------------------------------------------------------------
#                            GAME CONSTANTS
# -----------------------------------------------------------------------------
# These points are extracted from the "race_point" entries in big_g.json
# and manually sorted to create a continuous, logical racing line.
# They form the backbone for all mathematical calculations.
TRACK_NODES = [
    Vec3(2.28, 1.17, 6.02),   # Node 0: Near start line
    Vec3(4.85, 1.17, 6.04),   # Node 1
    Vec3(6.91, 1.17, 1.14),   # Node 2
    Vec3(2.68, 1.17, 0.77),   # Node 3
    Vec3(-0.38, 1.23, 1.92),  # Node 4: Lower-level near-center
    Vec3(-4.37, 1.17, -0.36), # Node 5: Start of ramp-up section
    Vec3(-7.54, 2.88, 3.29),  # Node 6: Top of ramp, upper-level
    Vec3(-7.63, 2.88, -3.62), # Node 7: Upper-level west side
    Vec3(-4.2, 2.88, -7.11),  # Node 8: Upper-level north curve
    Vec3(2.55, 2.88, -7.12),  # Node 9: Upper-level east side, before ramp-down
    Vec3(4.27, 2.2, -3.34),   # Node 10: Mid-point of ramp-down
    Vec3(0.41, 1.17, -3.39),  # Node 11: End of ramp-down
]

# -----------------------------------------------------------------------------
#                           CUSTOM GAME CLASSES
# -----------------------------------------------------------------------------

class Player(bs.Player['Team']):
    """Custom player class to store relay-specific data."""
    def __init__(self) -> None:
        self.relay_index: int = 0
        self.is_active_runner: bool = False
        self.spawn_pos: Sequence[float] = (0, 10, 0)
        self.marker_light: Optional[bs.Node] = None
        self.marker_ring: Optional[bs.Node] = None

class Team(bs.Team[Player]):
    """Custom team class to manage relay state."""
    def __init__(self) -> None:
        self.score: int = 0
        self.current_runner_index: int = 0
        self.lap_count: int = 0
        self.is_finished: bool = False
        self.finish_time: Optional[float] = None

# ba_meta export bascenev1.GameActivity
class RelayRaceGame(bs.TeamGameActivity[Player, Team]):
    """
    Big G Relay Race Game Activity.
    """

    name = 'Relay Race (Big G)'
    description = 'Pass the baton! Run to your teammate.'

    available_settings = [
        bs.IntSetting('Laps', min_value=1, default=1, increment=1),
        bs.BoolSetting('Disable Punching', default=True),
        bs.BoolSetting('Epic Mode', default=False),
    ]

    scoreconfig = bs.ScoreConfig(label='Time',
                                 scoretype=bs.ScoreType.MILLISECONDS,
                                 lower_is_better=True)

    @classmethod
    def supports_session_type(cls, sessiontype: type[bs.Session]) -> bool:
        return (
            issubclass(sessiontype, bs.DualTeamSession)
            or issubclass(sessiontype, bs.MultiTeamSession)
            or issubclass(sessiontype, bs.CoopSession)
        )

    @classmethod
    def get_supported_maps(cls, sessiontype: type[bs.Session]) -> List[str]:
        return ['Big G']

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._epic_mode = bool(settings.get('Epic Mode', False))
        self._total_laps = int(settings.get('Laps', 1))
        self._disable_punching = bool(settings.get('Disable Punching', True))

        self._race_started = False
        self._start_time: Optional[float] = None
        self._track_length: float = 0.0
        self._segment_lengths: List[float] = []

        self._pass_sound = bs.getsound('powerup01')
        self._win_sound = bs.getsound('score')
        self._chant_sound = bs.getsound('crowdChant')
        self._whistle_sound = bs.getsound('refWhistle')
        self._countdown_sound = bs.getsound('raceBeep1')
        self._go_sound = bs.getsound('raceBeep2')

        self.slow_motion = self._epic_mode
        self.default_music = bs.MusicType.RACE

    def on_begin(self) -> None:
        super().on_begin()
        self._calculate_track_geometry()
        self._spawn_teams()
        self._create_finish_line_visuals()
        bs.timer(1.5, self._start_countdown)
        bs.timer(0.2, self._update_race_status, repeat=True)

    def _calculate_track_geometry(self) -> None:
        """Calculates total track length by summing distances between nodes."""
        self._segment_lengths.clear()
        total_len = 0.0
        for i in range(len(TRACK_NODES)):
            p1 = TRACK_NODES[i]
            p2 = TRACK_NODES[(i + 1) % len(TRACK_NODES)]
            dist = (p2 - p1).length()
            self._segment_lengths.append(dist)
            total_len += dist
        self._track_length = total_len
        bs.print(f"Track length calculated: {self._track_length:.2f} units.")

    def _get_position_at_distance(self, distance: float, lane_offset: float = 0.0) -> Vec3:
        """
        Returns an exact (x, y, z) coordinate on the track for a given distance.
        This uses linear interpolation between the track nodes.
        """
        dist_on_lap = distance % self._track_length
        accumulated_dist = 0.0

        for i in range(len(TRACK_NODES)):
            seg_len = self._segment_lengths[i]
            if accumulated_dist + seg_len >= dist_on_lap:
                p1 = TRACK_NODES[i]
                p2 = TRACK_NODES[(i + 1) % len(TRACK_NODES)]
                remaining = dist_on_lap - accumulated_dist
                ratio = remaining / seg_len if seg_len > 0 else 0
                vector = p2 - p1
                pos = p1 + (vector * ratio)

                up_vector = Vec3(0, 1, 0)
                tangent = vector.normalize()
                right_vector = tangent.cross(up_vector).normalize()

                final_pos = pos + (right_vector * lane_offset)
                final_pos.y += 0.2  # Small vertical offset to prevent clipping
                return final_pos
            accumulated_dist += seg_len
        return TRACK_NODES[0]

    def _spawn_teams(self) -> None:
        """Calculates spawn points and spawns all players."""
        lane_width = 1.2

        for team_idx, team in enumerate(self.teams):
            player_count = len(team.players)
            if player_count == 0:
                continue

            offset = (team_idx - 0.5 * (len(self.teams) - 1)) * lane_width
            dist_per_player = self._track_length / player_count

            for i, player in enumerate(team.players):
                player.relay_index = i
                start_dist = i * dist_per_player
                pos = self._get_position_at_distance(start_dist, offset)
                player.spawn_pos = pos.to_tuple()

                self.spawn_player(player)

                if i == 0:
                    player.is_active_runner = True
                else:
                    player.is_active_runner = False
                    self._freeze_player_and_add_marker(player)

    def spawn_player(self, player: Player) -> bs.Actor:
        spaz = self.spawn_player_spaz(player, position=player.spawn_pos)
        spaz.connect_controls_to_player(enable_punch=not self._disable_punching,
                                        enable_bomb=False,
                                        enable_pickup=False)
        return spaz

    def _freeze_player_and_add_marker(self, player: Player) -> None:
        """Makes a player immobile and adds a visual marker."""
        if player.actor and player.actor.node:
            player.actor.node.frozen = True
            player.actor.node.invincible = True

        color = player.team.color
        pos = player.spawn_pos
        player.marker_ring = bs.newnode('locator', attrs={
            'shape': 'circle',
            'position': pos,
            'color': color,
            'opacity': 0.4,
            'size': [2.5],
            'draw_beauty': True,
            'additive': False
        })
        player.marker_light = bs.newnode('light', attrs={
            'position': pos,
            'color': color,
            'radius': 0.3,
            'intensity': 0.6
        })

    def _unfreeze_player_and_remove_marker(self, player: Player) -> None:
        """Activates a player and cleans up their waiting visuals."""
        if player.actor and player.actor.node:
            player.actor.node.frozen = False
            player.actor.node.invincible = False
            bs.getsound('shieldDown').play(position=player.actor.node.position)
            player.actor.node.handlemessage(bs.PowerupMessage('speed'))

        if player.marker_ring:
            player.marker_ring.delete()
            player.marker_ring = None
        if player.marker_light:
            player.marker_light.delete()
            player.marker_light = None

    def _create_finish_line_visuals(self) -> None:
        """Creates a visual marker for the start/finish line."""
        pos = TRACK_NODES[0]
        bs.newnode('locator', attrs={
            'shape': 'box',
            'position': (pos.x, pos.y + 1, pos.z),
            'size': [0.2, 4, 10],
            'color': (1, 1, 1),
            'opacity': 0.5,
            'draw_beauty': True
        })

    def _start_countdown(self) -> None:
        bs.timer(0.5, babase.Call(self._show_countdown_text, '3'))
        bs.timer(1.5, babase.Call(self._show_countdown_text, '2'))
        bs.timer(2.5, babase.Call(self._show_countdown_text, '1'))
        bs.timer(3.5, self._start_race)

    def _show_countdown_text(self, text: str) -> None:
        self._countdown_sound.play()
        node = bs.newnode('text', attrs={
            'text': text, 'v_attach': 'center', 'h_align': 'center',
            'scale': 2.0, 'color': (1, 1, 0.5), 'shadow': 1.0
        })
        bs.animate(node, 'scale', {0: 0, 0.2: 2.5, 0.8: 2.0, 1.0: 0})
        bs.timer(1.0, node.delete)

    def _start_race(self) -> None:
        self._go_sound.play()
        self._show_countdown_text('GO!')
        self._race_started = True
        self._start_time = bs.time()

    def _update_race_status(self) -> None:
        """The main game loop for checking handoffs and lap completion."""
        if not self._race_started or self.has_ended():
            return

        for team in self.teams:
            if team.is_finished:
                continue

            player_count = len(team.players)
            if player_count == 0:
                continue

            runner_idx = team.current_runner_index
            runner = team.players[runner_idx]

            if not runner.is_alive():
                continue

            runner_pos = Vec3(*runner.actor.node.position)

            # Check for handoff
            next_runner_idx = (runner_idx + 1) % player_count
            next_runner = team.players[next_runner_idx]

            dist_to_next = (runner_pos - Vec3(*next_runner.spawn_pos)).length()
            if dist_to_next < 2.5:
                # Check if we've run enough distance to be near the next point
                # This prevents instantly passing baton at game start.
                if team.lap_count > 0 or runner_idx > 0 or (runner_pos - Vec3(*runner.spawn_pos)).length() > 10:
                    self._handle_handoff(team, runner, next_runner)

    def _handle_handoff(self, team: Team, old_runner: Player, new_runner: Player) -> None:
        self._pass_sound.play()
        bs.emitfx(position=new_runner.actor.node.position, count=30, scale=1.5, spread=0.5, chunk_type='spark')
        bs.broadcastmessage(f"{new_runner.getname(full=True)} takes the baton for Team {team.id+1}!", color=team.color)

        # Deactivate old runner, activate new one
        old_runner.is_active_runner = False
        new_runner.is_active_runner = True

        # Despawn the old runner and respawn them at their original spot, frozen.
        old_runner.actor.handlemessage(bs.DieMessage())
        self.spawn_player(old_runner)
        self._freeze_player_and_add_marker(old_runner)

        # Unfreeze the new runner
        self._unfreeze_player_and_remove_marker(new_runner)

        # Check for lap completion
        if new_runner.relay_index == 0:
            team.lap_count += 1
            bs.broadcastmessage(f"Team {team.id+1} completed Lap {team.lap_count}!", color=team.color)
            if team.lap_count >= self._total_laps:
                self._team_finished(team)
                return

        team.current_runner_index = new_runner.relay_index

    def _team_finished(self, team: Team) -> None:
        if team.is_finished:
            return

        team.is_finished = True
        team.finish_time = bs.time() - self._start_time
        self._win_sound.play()
        self.stats.score(team, 1, screenmessage=False)

        bs.broadcastmessage(f"Team {team.id+1} finished the race in {team.finish_time:.2f} seconds!", color=team.color, scale=1.5)

        # End the game if all teams have finished
        if all(t.is_finished for t in self.teams):
            self.end_game()

    def handlemessage(self, msg: Any) -> Any:
        if isinstance(msg, bs.PlayerDiedMessage):
            player = msg.getplayer(Player)
            # Respawn players instantly at their designated spot if they die.
            if self._race_started and not player.team.is_finished:
                self.spawn_player(player)
                if player.is_active_runner:
                    self._unfreeze_player_and_remove_marker(player)
                else:
                    self._freeze_player_and_add_marker(player)
            # Don't call super, as we handle respawning differently.
        else:
            super().handlemessage(msg)

    def end_game(self) -> None:
        if self.has_ended():
            return

        results = bs.GameResults()
        for team in self.teams:
            if team.finish_time is not None:
                # Score is time in milliseconds
                results.set_team_score(team, int(team.finish_time * 1000))

        self.end(results=results)
