from typing import Optional, List

import math
import cv2
import supervision as sv
import numpy as np

from tacticalboard.configs.soccer import SoccerPitchConfiguration

def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    dark_background_color: sv.Color = sv.Color(30, 120, 30),
    line_color: sv.Color = sv.Color.WHITE,
    goal_color: sv.Color = sv.Color(200, 200, 200),
    padding: int = 50,
    line_thickness: int = 4,
    goal_line_thickness: int = 6,
    point_radius: int = 8,
    scale: float = 0.1,
    num_stripes: int = 12
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, scale,
    goals, arcs, and grass pattern.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)
    scaled_corner_arc_radius = int(config.corner_arc_radius * scale)
    scaled_penalty_arc_radius = int(config.penalty_arc_radius * scale)
    scaled_goal_width = int(config.goal_width * scale)
    scaled_goal_depth = int(config.goal_depth * scale)

    pitch_only_image = np.zeros((scaled_width, scaled_length, 3), dtype=np.uint8)
    
    stripe_width = scaled_length / num_stripes
    for i in range(num_stripes):
        x_start = int(i * stripe_width)
        x_end = int((i + 1) * stripe_width)
        color = dark_background_color if i % 2 == 0 else background_color
        cv2.rectangle(pitch_only_image, 
                      (x_start, 0), 
                      (x_end, scaled_width), 
                      color.as_bgr(), 
                      thickness=-1)

    final_image_height = scaled_width + 2 * padding
    final_image_width = scaled_length + 2 * padding
    pitch_image = np.ones((final_image_height, final_image_width, 3), dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)
    
    pitch_image[padding:padding+scaled_width, padding:padding+scaled_length] = pitch_only_image

    for start, end in config.edges:
        x1 = config.vertices[start - 1][0] * scale + padding
        y1 = config.vertices[start - 1][1] * scale + padding
        x2 = config.vertices[end - 1][0] * scale + padding
        y2 = config.vertices[end - 1][1] * scale + padding
        
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center_x = scaled_length // 2 + padding
    centre_circle_center_y = scaled_width // 2 + padding
    cv2.circle(
        img=pitch_image,
        center=(centre_circle_center_x, centre_circle_center_y),
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
            thickness=line_thickness
        )
    
    penalty_spots_coords = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot_x, spot_y in penalty_spots_coords:
        cv2.circle(
            img=pitch_image,
            center=(int(spot_x), int(spot_y)),
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    corners = [
        (padding, padding),
        (padding, scaled_width + padding),
        (scaled_length + padding, padding),
        (scaled_length + padding, scaled_width + padding)
    ]
    corner_angles = [
        (0, 90),
        (270, 360),
        (90, 180),
        (180, 270)
    ]
    for (cx, cy), (start_angle, end_angle) in zip(corners, corner_angles):
        cv2.ellipse(
            img=pitch_image,
            center=(cx, cy),
            axes=(scaled_corner_arc_radius, scaled_corner_arc_radius),
            angle=0,
            startAngle=start_angle,
            endAngle=end_angle,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
        
    for spot_x, spot_y in penalty_spots_coords:
        center_x = int(spot_x)
        center_y = int(spot_y)
        
        dx = abs(config.penalty_box_length * scale - scaled_penalty_spot_distance)
        if scaled_penalty_arc_radius**2 > dx**2:
            dy = math.sqrt(scaled_penalty_arc_radius**2 - dx**2)
            alpha_rad = math.asin(dy / scaled_penalty_arc_radius)
            alpha_deg = math.degrees(alpha_rad)
            
            if center_x < final_image_width / 2:
                start_angle = -alpha_deg
                end_angle = alpha_deg
            else:
                start_angle = 180 - alpha_deg
                end_angle = 180 + alpha_deg

            cv2.ellipse(
                img=pitch_image,
                center=(center_x, center_y),
                axes=(scaled_penalty_arc_radius, scaled_penalty_arc_radius),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=line_color.as_bgr(),
                thickness=line_thickness
            )

    goal_y1 = scaled_width // 2 + padding - scaled_goal_width // 2
    goal_y2 = scaled_width // 2 + padding + scaled_goal_width // 2
    
    goal_posts_left = [
        (padding, goal_y1),
        (padding, goal_y2),
        (padding - scaled_goal_depth, goal_y1),
        (padding - scaled_goal_depth, goal_y2)
    ]
    cv2.line(pitch_image, goal_posts_left[0], goal_posts_left[2], goal_color.as_bgr(), goal_line_thickness)
    cv2.line(pitch_image, goal_posts_left[1], goal_posts_left[3], goal_color.as_bgr(), goal_line_thickness)
    cv2.line(pitch_image, goal_posts_left[2], goal_posts_left[3], goal_color.as_bgr(), goal_line_thickness)
    
    goal_posts_right = [
        (scaled_length + padding, goal_y1),
        (scaled_length + padding, goal_y2),
        (scaled_length + padding + scaled_goal_depth, goal_y1),
        (scaled_length + padding + scaled_goal_depth, goal_y2)
    ]
    cv2.line(pitch_image, goal_posts_right[0], goal_posts_right[2], goal_color.as_bgr(), goal_line_thickness)
    cv2.line(pitch_image, goal_posts_right[1], goal_posts_right[3], goal_color.as_bgr(), goal_line_thickness)
    cv2.line(pitch_image, goal_posts_right[2], goal_posts_right[3], goal_color.as_bgr(), goal_line_thickness)
    
    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch

def draw_player_on_pitch(
    config: SoccerPitchConfiguration,
    xy: tuple[float, float],
    jersey_number: int = 1,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    display_text: Optional[str] = None
) -> np.ndarray:
    """
    Draws players on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        jersey_number (int, optional): Jersey number of the player.
            Defaults to 1.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.
        display_text (Optional[str], optional): Custom text to display instead of jersey_number.
            If None, jersey_number will be used. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )
    
    scaled_point = (
        int(xy[0] * scale) + padding,
        int(xy[1] * scale) + padding
    )
    
    text_to_display = display_text if display_text is not None else str(jersey_number)
    
    text_len = len(text_to_display)
    if text_len <= 1:
        alignment_offset = 5
        font_scale = 0.6
    elif text_len == 2:
        alignment_offset = 10
        font_scale = 0.5
    else:
        alignment_offset = 12
        font_scale = 0.4
        
    cv2.circle(
        img=pitch,
        center=scaled_point,
        radius=radius,
        color=face_color.as_bgr(),
        thickness=-1
    )
    cv2.putText(
        img=pitch, 
        text=text_to_display, 
        org=(scaled_point[0] - alignment_offset, scaled_point[1] + 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=font_scale, 
        color=(255, 255, 255),
        thickness=1)
    cv2.circle(
        img=pitch,
        center=scaled_point,
        radius=radius,
        color=edge_color.as_bgr(),
        thickness=thickness
    )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 1.
        team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 2.
        team_1_color (sv.Color, optional): Color representing the control area of
            team 1. Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color representing the control area of
            team 2. Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the Voronoi diagram overlay.
            Defaults to 0.5.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay
