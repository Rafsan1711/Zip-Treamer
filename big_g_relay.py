# ba_meta require api 9
# -----------------------------------------------------------------------------
#
# Plugin Name: Big G Relay Race
# Description: A mathematical relay race mode for the Big G map.
#              Calculates spawn points along the track vector curve based on
#              player count. Supports multi-team and single-team play.
#
# Author: Rafsan
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
# Metadata for Plugin Manager
# -----------------------------------------------------------------------------
plugman = {
    "plugin_name": "big_g_relay",
    "description": "Relay race on Big G where track positions are calculated mathematically.",
    "authors": [
        {"name": "Rafsan", "email": "rafsan@example.com"}
    ],
    "version": "1.0.0",
}

# -----------------------------------------------------------------------------
# Vector Math Helper Class
# -----------------------------------------------------------------------------
class Vec3:
    """A helper class for 3D vector calculations to determine track positions."""
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
        if l == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def cross(self, other: Vec3) -> Vec3:
        """Cross product for calculating lane offsets."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Define the Big G track path nodes.
# These coordinates represent the center line of the drivable area.
# Ordered: Start(Bottom Right) -> Bottom Loop -> Ramp Up -> Top Loop -> Ramp Down
TRACK_NODES = [
    Vec3(6.91, 1.17, 1.14),    # Node 0: Start Line (Bottom Straight East)
    Vec3(4.85, 1.17, 6.04),    # Node 1: Bottom Curve Start
    Vec3(-0.38, 1.23, 7.00),   # Node 2: Bottom Curve Apex (Approx from visual)
    Vec3(-5.00, 1.17, 5.00),   # Node 3: Bottom Curve End
    Vec3(-7.54, 2.00, 0.00),   # Node 4: Ramp Up Midpoint
    Vec3(-7.63, 2.88, -3.62),  # Node 5: Top Curve Start (West)
    Vec3(-4.2, 2.88, -7.11),   # Node 6: Top Curve Mid
    Vec3(2.55, 2.88, -7.12),   # Node 7: Top Curve End / Top Straight
    Vec3(7.00, 2.00, -3.34),   # Node 8: Ramp Down
]

# We close the loop by connecting Node 8 back to Node 0 automatically in logic.

# -----------------------------------------------------------------------------
# Game Classes
# -----------------------------------------------------------------------------

class Player(bs.Player['Team']):
    """Our player type for this game."""
    def __init__(self) -> None:
        self.relay_index: int = 0
        self.distance_target: float = 0.0
        self.spawn_pos: Sequence[float] = (0, 10, 0)
        self.has_run: bool = False
        self.is_active_runner: bool = False
        self.marker_light: Optional[bs.Node] = None
        self.marker_ring: Optional[bs.Node] = None

class Team(bs.Team[Player]):
    """Our team type for this game."""
    def __init__(self) -> None:
        self.current_runner_index: int = 0
        self.lap_count: int = 0
        self.finished: bool = False
        self.baton_pass_count: int = 0

# ba_meta export bascenev1.GameActivity
class RelayRaceGame(bs.TeamGameActivity[Player, Team]):
    """
    Big G Relay Race Game.
    Calculates spawn positions mathematically along the track path.
    Scales laps based on player count.
    """

    name = 'Big G Relay Race'
    description = 'Pass the baton! Run to your teammate.'
    
    # Available settings
    available_settings = [
        bs.BoolSetting('Epic Mode', default=False),
        bs.BoolSetting('Disable Punching', default=True),
        bs.IntChoiceSetting(
            'Laps Per Player Scale',
            choices=[
                ('Short (0.5 Lap/Player)', 1),
                ('Normal (1.0 Lap/Player)', 2),
                ('Long (2.0 Lap/Player)', 3),
            ],
            default=2,
        ),
    ]
    
    scoreconfig = bs.ScoreConfig(label='Time',
                                 scoretype=bs.ScoreType.MILLISECONDS,
                                 lower_is_better=True)

    @classmethod
    def supports_session_type(cls, sessiontype: type[bs.Session]) -> bool:
        # Supports both Teams and Co-op (2 players in 1 team can play)
        return issubclass(sessiontype, bs.DualTeamSession) or issubclass(sessiontype, bs.MultiTeamSession) or issubclass(sessiontype, bs.CoopSession)

    @classmethod
    def get_supported_maps(cls, sessiontype: type[bs.Session]) -> List[str]:
        return ['Big G']

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._score_to_win = 1
        self._epic_mode = bool(settings.get('Epic Mode', False))
        self._scale_setting = int(settings.get('Laps Per Player Scale', 2))
        self._disable_punching = bool(settings.get('Disable Punching', True))
        
        self._race_started = False
        self._track_length = 0.0
        self._segment_lengths: List[float] = []
        self._total_laps = 1
        
        # Audio
        self._pass_sound = bs.getsound('powerup01')
        self._win_sound = bs.getsound('score')
        self._chant_sound = bs.getsound('crowdChant')
        self._whistle_sound = bs.getsound('refWhistle')

        # Base class overrides
        self.slow_motion = self._epic_mode
        self.default_music = (bs.MusicType.EPIC if self._epic_mode else
                              bs.MusicType.RACE)

    def on_begin(self) -> None:
        super().on_begin()
        
        # 1. Analyze Track Geometry
        self._calculate_track_geometry()
        
        # 2. Setup Game Logic based on Player Count
        self._setup_race_logic()
        
        # 3. Spawn Players at calculated positions
        self._spawn_teams()
        
        # 4. Create Finish Line Visuals
        self._create_finish_line()
        
        # 5. Start Countdown
        bs.timer(1.0, self._start_countdown)
        
        # 6. Start Update Loop for Baton Passing
        bs.timer(0.1, self._update_race_status, repeat=True)

    def _calculate_track_geometry(self) -> None:
        """
        Mathematically calculates the total length of the Big G track 
        by summing distances between defined nodes.
        """
        self._segment_lengths = []
        total_len = 0.0
        
        # Loop through nodes to form a closed circuit
        for i in range(len(TRACK_NODES)):
            p1 = TRACK_NODES[i]
            p2 = TRACK_NODES[(i + 1) % len(TRACK_NODES)] # Wrap back to 0
            
            dist = (p2 - p1).length()
            self._segment_lengths.append(dist)
            total_len += dist
            
        self._track_length = total_len
        bs.print_level_data(f"Track Analysis Complete. Total Length: {self._track_length:.2f} game units.")

    def _get_position_at_distance(self, distance: float, lane_offset: float = 0.0) -> Vec3:
        """
        Returns the exact (x,y,z) coordinate on the track for a given distance 
        from the start line. Handles wrapping around laps.
        
        Args:
            distance: Distance traveled from start node.
            lane_offset: Perpendicular offset for different teams (lanes).
        """
        # Normalize distance to one lap
        dist_on_lap = distance % self._track_length
        
        accumulated_dist = 0.0
        
        for i in range(len(TRACK_NODES)):
            seg_len = self._segment_lengths[i]
            
            # Check if the target point is in this segment
            if accumulated_dist + seg_len >= dist_on_lap:
                # Found the segment
                p1 = TRACK_NODES[i]
                p2 = TRACK_NODES[(i + 1) % len(TRACK_NODES)]
                
                # Linear Interpolation (Lerp) ratio
                remaining = dist_on_lap - accumulated_dist
                ratio = remaining / seg_len
                
                # Base position on the center line
                vector = p2 - p1
                pos = p1 + (vector * ratio)
                
                # Calculate Lane Offset (Perpendicular vector)
                # Up vector assumed (0, 1, 0)
                up = Vec3(0, 1, 0)
                tangent = vector.normalize()
                right = tangent.cross(up).normalize()
                
                # Apply lane offset
                final_pos = pos + (right * lane_offset)
                
                # Raycast down to find ground Y (safety check to ensure feet on floor)
                # Since we calculated Y manually, we trust the nodes, 
                # but adding a small Y bump prevents clipping.
                final_pos.y += 0.5 
                
                return final_pos
            
            accumulated_dist += seg_len
            
        return TRACK_NODES[0] # Fallback

    def _setup_race_logic(self) -> None:
        """Determines lap count based on player count."""
        max_team_size = 0
        for team in self.teams:
            max_team_size = max(max_team_size, len(team.players))
            
        if max_team_size == 0:
            return

        # LOGIC:
        # Scale 1 (Short): 2 players = 1 lap total.
        # Scale 2 (Normal): 1 player = 1 lap. 2 players = 2 laps.
        
        if self._scale_setting == 1:
            self._total_laps = max(1, math.ceil(max_team_size * 0.5))
        elif self._scale_setting == 2:
            self._total_laps = max(1, max_team_size)
        else:
            self._total_laps = max(2, max_team_size * 2)
            
        # Time limit adjustment
        self.setup_standard_time_limit(int(self._total_laps * 45)) 

        bs.broadcastmessage(f"Race Config: {self._total_laps} Laps | Team Size: {max_team_size}", color=(0,1,1))

    def _spawn_teams(self) -> None:
        """
        Spawns players at their calculated relay points.
        """
        lane_width = 1.5 # Distance from center line
        
        for team_idx, team in enumerate(self.teams):
            player_count = len(team.players)
            if player_count == 0:
                continue
                
            # Calculate lane offset based on team index
            # Team 0: -0.75, Team 1: +0.75
            offset = -0.75 if team_idx % 2 == 0 else 0.75
            if len(self.teams) > 2:
                offset = (team_idx - (len(self.teams)/2)) * 1.0

            # Calculate the total distance the team needs to run
            total_race_distance = self._track_length * self._total_laps
            
            # Segment per player
            dist_per_player = total_race_distance / player_count
            
            for i, player in enumerate(team.players):
                player.relay_index = i
                
                # The starting distance for this player
                # Player 0 starts at 0.
                # Player 1 starts at dist_per_player.
                start_dist = i * dist_per_player
                
                player.distance_target = start_dist
                
                # Calculate exact 3D coordinate
                pos = self._get_position_at_distance(start_dist, offset)
                player.spawn_pos = pos.to_tuple()
                
                self.spawn_player(player)
                
                # Configure Initial State
                if i == 0:
                    player.is_active_runner = True
                    self._highlight_runner(player)
                else:
                    player.is_active_runner = False
                    self._freeze_player(player)
                    self._create_waiting_marker(player)

    def spawn_player(self, player: Player) -> bs.Actor:
        spaz = self.spawn_player_spaz(player, position=player.spawn_pos)
        
        # Reset specific attributes
        spaz.connect_controls_to_player(enable_punch=not self._disable_punching,
                                        enable_bomb=False,
                                        enable_pickup=False,
                                        enable_jump=True,
                                        enable_run=True)
                                        
        # If frozen (waiting for baton)
        if not player.is_active_runner:
            spaz.node.invincible = True
            spaz.node.frozen = True 
            # Force facing direction? BombSquad spaz faces movement direction naturally.
            
        return spaz

    def _freeze_player(self, player: Player) -> None:
        if player.actor and player.actor.node:
            player.actor.node.frozen = True
            player.actor.node.invincible = True

    def _unfreeze_player(self, player: Player) -> None:
        if player.actor and player.actor.node:
            player.actor.node.frozen = False
            player.actor.node.invincible = False
            bs.getsound('shieldDown').play(position=player.actor.node.position)
            # Give a small speed boost on tag
            player.actor.node.handlemessage("impulse", 0,0,0, 0,0,0, 200, 200, 0, 0, 0, 100, 0)

    def _create_waiting_marker(self, player: Player) -> None:
        """Visual circle for where the player is waiting."""
        if player.marker_ring:
            player.marker_ring.delete()
            
        color = player.team.color
        player.marker_ring = bs.newnode('locator', attrs={
            'shape': 'circle',
            'position': player.spawn_pos,
            'color': color,
            'opacity': 0.5,
            'size': [2.0],
            'draw_beauty': True,
            'additive': True
        })
        # Add a light
        player.marker_light = bs.newnode('light', attrs={
            'position': player.spawn_pos,
            'color': color,
            'radius': 0.2,
            'intensity': 0.5
        })

    def _highlight_runner(self, player: Player) -> None:
        """Put a star or light over the active runner."""
        if player.actor and player.actor.node:
            # We can attach a light to the spaz
            light = bs.newnode('light', attrs={
                'color': (1, 1, 1),
                'radius': 0.1,
                'intensity': 1.0,
                'height_attenuated': False
            })
            player.actor.node.connectattr('position', light, 'position')
            # Auto delete light after some time or track it? 
            # For simplicity, we just let it be part of the actor or let it die with actor respawn.
            bs.animate(light, 'intensity', {0:0.5, 0.5:1.0, 1.0:0.5}, loop=True)

    def _create_finish_line(self) -> None:
        """Visual finish line at Node 0."""
        pos = TRACK_NODES[0]
        # Create a region for win detection (though we calculate distance mathematically mainly)
        # But visual guide is good.
        bs.newnode('locator', attrs={
            'shape': 'box',
            'position': (pos.x, pos.y + 1, pos.z),
            'size': [2, 5, 8], # Wide across the track
            'color': (1, 1, 1),
            'opacity': 0.3,
            'draw_beauty': True
        })
        
        # Checkered texture effect simulation (particles)
        for i in range(-3, 4):
            bs.emitfx(position=(pos.x, pos.y+0.5, pos.z + i),
                      velocity=(0, 2, 0),
                      count=5,
                      scale=1.0,
                      spread=0.1,
                      chunk_type='spark')

    def _start_countdown(self) -> None:
        bs.getsound('tick').play()
        self._countdown_text(3)
        bs.timer(1.0, lambda: self._countdown_text(2))
        bs.timer(2.0, lambda: self._countdown_text(1))
        bs.timer(3.0, self._start_race_actual)

    def _countdown_text(self, num: int) -> None:
        if num > 0:
            bs.getsound('tick').play()
            color = (1, 0.5, 0)
        else:
            color = (0, 1, 0)
            
        t = bs.newnode('text', attrs={
            'text': str(num) if num > 0 else "GO!",
            'scale': 2.0,
            'position': (0, 100),
            'h_align': 'center',
            'v_attach': 'center',
            'color': color,
            'shadow': 1.0,
            'lifespan': 1000
        })
        bs.animate(t, 'scale', {0: 2.0, 0.5: 0.0})

    def _start_race_actual(self) -> None:
        self._countdown_text(0)
        self._whistle_sound.play()
        self._race_started = True

    def _update_race_status(self) -> None:
        """
        Main loop to check for relay handoffs and win conditions.
        """
        if not self._race_started:
            return

        for team in self.teams:
            if team.finished:
                continue

            # Get current active runner
            current_idx = team.current_runner_index
            if current_idx >= len(team.players):
                continue
                
            runner = team.players[current_idx]
            
            if not runner.is_alive() or not runner.actor or not runner.actor.node:
                # If runner died, we should probably respawn them at their start point
                # But bascenev1 handles auto respawn usually. 
                # We force logic if they respawned elsewhere?
                continue

            runner_pos = Vec3(runner.actor.node.position[0], runner.actor.node.position[1], runner.actor.node.position[2])

            # Check if this is the Last Runner
            is_last_runner = (current_idx == len(team.players) - 1)
            
            if is_last_runner:
                # Check for Finish Line Crossing
                # Finish line is Node 0. 
                # We verify if they are close to Node 0 AND have completed the lap distance approximation.
                # A simple distance check to start point might trigger prematurely if they turn around.
                # So we check if they are in the last segment of the track.
                
                # Simple check: distance to start node < 3.0 units
                dist_to_finish = (runner_pos - TRACK_NODES[0]).length()
                
                # To prevent triggering at start, check if they moved (not handled here but assumed)
                # Better logic: Check if they passed the last node (Node 8) heading to Node 0.
                dist_to_pre_finish = (runner_pos - TRACK_NODES[8]).length()
                
                if dist_to_finish < 4.0 and dist_to_pre_finish < 15.0:
                    # They likely finished
                    self._team_finished(team)
            
            else:
                # Check for Relay Handoff
                next_runner = team.players[current_idx + 1]
                
                # Distance to the next waiting player
                # We use the calculated spawn pos of the next player
                target_pos = Vec3(next_runner.spawn_pos[0], next_runner.spawn_pos[1], next_runner.spawn_pos[2])
                dist_to_target = (runner_pos - target_pos).length()
                
                # Threshold for passing baton (3.0 units)
                if dist_to_target < 3.0:
                    self._handle_relay_handoff(team, runner, next_runner)

    def _handle_relay_handoff(self, team: Team, old_runner: Player, new_runner: Player) -> None:
        """Executes the baton pass logic."""
        self._pass_sound.play()
        
        # Visual FX
        bs.emitfx(position=old_runner.actor.node.position,
                  velocity=(0, 5, 0), count=20, scale=2.0, spread=1.0, chunk_type='spark')
        
        # 1. Freeze/Remove Old Runner
        # We can just despawn them to clear the track, or freeze them.
        # Despawning is cleaner for relay.
        old_runner.actor.handlemessage(bs.DieMessage())
        
        # 2. Unfreeze New Runner
        new_runner.is_active_runner = True
        self._unfreeze_player(new_runner)
        self._highlight_runner(new_runner)
        
        # 3. Cleanup Markers
        if new_runner.marker_ring:
            new_runner.marker_ring.delete()
        if new_runner.marker_light:
            new_runner.marker_light.delete()
            
        # 4. Update Team State
        team.current_runner_index += 1
        
        # Broadcast
        bs.broadcastmessage(f"{new_runner.getname(full=True)} has the baton!", color=team.color)

    def _team_finished(self, team: Team) -> None:
        team.finished = True
        self._win_sound.play()
        bs.broadcastmessage(f"TEAM {team.name} WINS!", color=team.color, scale=2.0)
        
        # End game
        self.end_game()

    def end_game(self) -> None:
        results = bs.GameResults()
        for team in self.teams:
            if team.finished:
                # We can use time as score since lower is better in settings
                results.set_team_score(team, int(bs.time() * 1000))
            else:
                results.set_team_score(team, 0)
        self.end(results=results)
