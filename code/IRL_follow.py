import googlemaps
from onvif import ONVIFCamera
import shutil
import os
import requests

gmaps = googlemaps.Client(key='AIzaSyAgDXQNFfqeb5EjkF0f8BzdkDPebQCkj-A')

def snap_to_roads(api_key, path):
    url = "https://roads.googleapis.com/v1/snapToRoads"    #use google's Roads API
    params = {
        'path': '|'.join([f"{lat},{lng}" for lat, lng in path]),
        'interpolate': True,
        'key': api_key
    }
    response = requests.get(url, params=params)
    snapped_points = response.json().get('snappedPoints', [])
    return [(point['location']['latitude'], point['location']['longitude']) for point in snapped_points]


def get_next_intersection(api_key, current_position, direction, num_points=5, distance_per_point=0.0001):
    # num_points is the Number of points to snap along the road.
    # distance_per_point is the Approximate distance between consecutive points

    lat, lng = current_position
    path = [(lat, lng)]
    
    for i in range(1, num_points + 1):
        if direction == 'up':
            lat += distance_per_point
        elif direction == 'down':
            lat -= distance_per_point
        elif direction == 'left':
            lng -= distance_per_point
        elif direction == 'right':
            lng += distance_per_point
        
        path.append((lat, lng))
    
    snapped_points = snap_to_roads(api_key, path)
    
    next_intersection = snapped_points[-1]  # Assuming the last snapped point is close to the next intersection
    
    return next_intersection

def activate_camera(ip, port, username, password, find_this_folder, prev_coordinates):
    mycam = ONVIFCamera(ip, port, username, password)
    
    media_service = mycam.create_media_service()
    ptz_service = mycam.create_ptz_service()
    
    ptz_configuration = ptz_service.GetConfigurations()[0]
    request = ptz_service.create_type('ContinuousMove')
    request.ProfileToken = media_service.GetProfiles()[0]._token
    request.Velocity = ptz_service.GetStatus({'ProfileToken': ptz_configuration.token}).Position
    request.Velocity.PanTilt._x = 0.1  
    ptz_service.ContinuousMove(request)
    
    print(f"Camera at {ip} activated using ONVIF.")
    # Maybe have some extra HTTPS requests here
    
    #transfer_find_this_folder(ip, port, username, password, find_this_folder)     # Too complex for now, prob sedn trhough a server

    #send_previous_coordinates(ip, port, username, password, prev_coordinates)     # Through an HTTPS requests

def follow_car(starting_coords, final_directions):
    current_position = starting_coords

    car_positions = [current_position]

    for direction in final_directions:
        # Unless the vehicle goes offroad, we go the next possible intersection(s) to find the next position of the car
        next_position = get_next_intersection(gmaps, current_position, direction)   
        car_positions.append(next_position)
        current_position = next_position  # Update current position
        print(f"Car moved {direction}, new position: {current_position}")

    previous_cameras_coords = []     # This would probably be given by the camera SDKs and/or HTTPS requests
    # I'm guessing we use google maps to find the nearby cameras through the coordiantes, and then conenct to a server to get all the rest of the info
    next_camera_coords_list = [(37.7750, -122.4195), (37.7745, -122.4185)]    
    # Example info of the next cameras (I don't know how to get this data, it would have to be connected on a server)
    next_camera_ips_list = []
    next_camera_ports_list = []
    username = "peppapig129"   # Example values, again I would need to be an actual operator to have this info
    password = "12345678910"   
    find_this_folder = "find_this"    # I know its unnecessary to have this here, but this is for formatting adn future editors' purposes

    for i in range(len(next_camera_coords_list)):
        print(f"Activating camera. IP is {next_camera_ips_list[i]}")
        prev_coordinates = previous_cameras_coords.copy()
        prev_coordinates.append(next_camera_coords_list[i])
        # Activate the camera (assume the signal is sent and then the camera operates by itself so that this for loops can iterate quickly)
        activate_camera(next_camera_ips_list[i], next_camera_ports_list[i], username, password, find_this_folder, prev_coordinates)

    
