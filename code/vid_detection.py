import cv2
import time
import numpy as np
import os, json
from sklearn.cluster import KMeans
import easyocr, imutils
from matplotlib import pyplot as plt

global first_camera_starting_coords 
first_camera = True     #this wil be given by user or the other cameras in the sequence     
if first_camera:
    lat = float(input('Enter the latitude of your camera: '))   # for example, 37.7749
    long = float(input('Enter the longitude of your camera: '))  # for example, -122.4194
    first_camera_starting_coords = (lat, long)     # for example, (37.7749, -122.4194)
    user_car_color = str(input('Enter the dominant color (red, blue, white, black) of your car. Red and white are the best ones right now: '))
    user_car_license_plate = str(input('Enter the license plate of your car: '))   #example values: 6AT 4838 
    user_images_done = input("Take a few pictures of your car from the camera's position and upload them into the user_car_images folder...(press enter after done)")
    start = False
    while not start:
        start_now = input('Do you want to start now? (y/n) ')     
        if start_now == 'y':
            start = True
    final_directions = []
else:
   #retrieve the first camera's starting coords from the Camera SDK & ONVIF
   print('not first camera.....get data from Camera SDK & ONVIF')
   first_camera_starting_coords = (37.7749, -122.4194)  #exampel values, change leter

# Initialize video capture. 0 for computer webcam. 
cap = cv2.VideoCapture('data/cars.mp4')    #For live feed, then give the IPaddress of the security camera(s)
# or yes_car.mp4, or no_car.mp4.
# Create a folder to save frames
output_folder = 'frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frame_count = 0

car_classifier = cv2.CascadeClassifier('data/haarcascade_cars.xml')
body_classifier = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')
bus_classifier = cv2.CascadeClassifier('data/Bus_front.xml')
bike_classifier = cv2.CascadeClassifier('data/two_wheeler.xml')
Lplate_classifier = cv2.CascadeClassifier('data/haarcascade_russian_plate_number.xml')

def get_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels)

    sorted_indices = np.argsort(percentages)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_percentages = percentages[sorted_indices]

    main_color = sorted_colors[0].astype(int)
    print(sorted_colors)
    print('First color:', color_name(main_color))
    if len(sorted_colors) > 1 and 0 <= main_color[0] <= 5 and 0 <= main_color[1] <= 5 and 0 <= main_color[2] <= 5:    #black, the empty space
        print('Second color:', color_name(sorted_colors[1].astype(int)))
        print('Third color:', color_name(sorted_colors[2].astype(int)))
        return sorted_colors[1].astype(int), sorted_percentages[1]     #next dominant color which is likely the car
    return sorted_colors[0].astype(int), sorted_percentages[0]     #if dominant color is not black

def color_name(bgr_color):
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

    color_ranges = {
        "Red": [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
        "Orange": [((11, 100, 100), (25, 255, 255))],
        "Yellow": [((26, 100, 100), (35, 255, 255))],
        "Green": [((36, 100, 100), (70, 255, 255))],
        "Blue": [((100, 100, 100), (130, 255, 255))],
        "Purple": [((131, 100, 100), (160, 255, 255))],
        "White": [((0, 0, 200), (180, 30, 255))],
        "Black": [((0, 0, 0), (180, 255, 30))],
        "Gray": [((0, 0, 31), (180, 30, 199))]
    }
    
    for name, ranges in color_ranges.items():
        for start, end in ranges:
            if np.all(hsv_color >= start) and np.all(hsv_color <= end):
                return name
    
    return "Unknown"

def detect_license_plate(img):
    
    def plt_show(image, title="", gray=False, size=(10, 10)):
        temp = image
        if not gray:
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=size)
        plt.title(title)
        plt.imshow(temp, cmap='gray' if gray else None)
        plt.show()

    def detect_number(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        number_plates = Lplate_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        #print("Number plates detected: " + str(len(number_plates)))
        if len(number_plates) == 0:
            return False
        else:
            for (x, y, w, h) in number_plates:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            return True
        
    plate_detected = detect_number(img)
    if not plate_detected:
        return None
    else:
        img = cv2.resize(img, (620, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        # edge detection
        edged = cv2.Canny(gray, 30, 200)

        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        # Check if a license plate contour was found
        if screenCnt is None:
            print("No contour detected")
            return None
        
        # Draw the contour on the image
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        # Mask the area outside the license plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Crop the region containing the license plate
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # Use EasyOCR to read the text from the license plate
        # Create an EasyOCR reader
        reader = easyocr.Reader(['en'])  # Use English language for OCR
        results = reader.readtext(Cropped)
        
        # Extract the detected text
        if results:
            text = " ".join([result[1] for result in results])
            print("Detected license plate number is:", text)
            return text
        else:
            #print("No text detected")
            return None

#Test out detect_license_plate
#img = cv2.imread('data/car_license_plate.png')
#detect_license_plate(img)

def store_car_info(color, license_plate, images, starting_coords, final_directions, folder="find_this"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    car_id = f"{color}_{license_plate}"    #could cause some confusion when tryign to match cars if color is unknown and license palte is None
    car_folder = os.path.join(folder, car_id)
    
    if not os.path.exists(car_folder):
        os.makedirs(car_folder)
    
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(car_folder, f"car_image{i}.jpg"), image)
    
    metadata = {
        "color": color,
        "license_plate": license_plate,
        "starting_coords": starting_coords,
        "final_directions": final_directions
    }
    with open(os.path.join(car_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def load_car_info(folder="find_this"):
    car_infos = []

    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return car_infos

    for car_folder in os.listdir(folder):
        car_path = os.path.join(folder, car_folder)
        if os.path.isdir(car_path):
            metadata_path = os.path.join(car_path, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
            images = []
            for image_name in os.listdir(car_path):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(car_path, image_name)
                    image = cv2.imread(image_path)
                    images.append(image)

            car_info = {
                "color": metadata["color"],
                "license_plate": metadata["license_plate"],
                "images": images,
                "starting_coords": metadata["starting_coords"],
                "final_directions": metadata["final_directions"]
            }
            car_infos.append(car_info)
    
    return car_infos

def detect_specific_car(frame, fgmask, forground, target_color, target_license_plate, target_images, threshold=0.52):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    buses = bus_classifier.detectMultiScale(gray, 1.4, 2)
    bikes = bike_classifier.detectMultiScale(gray, 1.4, 2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cars1 = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            cars1.append((x, y, w, h))
    
    run = True
    v = 0
    vehicles = [cars]  # Use either cars or cars1 depending on accuracy of cars1 (right now not very good, but in the future, it will be a better way)
    while run:
        for (x, y, w, h) in vehicles[v]:
            prob = 0
            car_crop = forground[y:y+h, x:x+w]      #use frame or forgorund dependign on video ------------------------

            color_name_detected, license_plate = detect_all(forground, car_crop, x, y, w, h, frame, frameCopy, (255, 0, 0))
            
            if color_name_detected.lower() == target_color.lower():
                print('Color accepted')
                prob += 0.3             #should be 0.3

                if target_license_plate:        #not None
                    if license_plate:      #not None
                        num = min(len(license_plate), len(target_license_plate))     #num will be 6 or 8
                        diff = abs(len(license_plate) - len(target_license_plate))/100
                        for i in range(num):
                            if license_plate[i] == target_license_plate[i]:
                                prob += 0.5/num - diff
                        print('License plate accepted')
                
                average_vals = []
                #cv2.imwrite("car_image.jpg", car_crop)
                for target_image in target_images:
                    # Check if target image and car_crop have the same number of channels
                    if car_crop.shape[-1] != target_image.shape[-1]:
                        print(f"Channel mismatch: car_crop has {car_crop.shape[-1]} channels, target_image has {target_image.shape[-1]} channels")
                        continue
                    
                    # Resize target image if it's larger than the car_crop
                    if car_crop.shape[0] < target_image.shape[0] or car_crop.shape[1] < target_image.shape[1]:
                        scale_factor = min(car_crop.shape[0] / target_image.shape[0], car_crop.shape[1] / target_image.shape[1])
                        resized_target_image = cv2.resize(target_image, (0, 0), fx=scale_factor, fy=scale_factor)
                        print(f"Resized target_image to {resized_target_image.shape}")
                    else:
                        resized_target_image = target_image
                    
                    # Perform the match
                    result1 = cv2.matchTemplate(car_crop, resized_target_image, cv2.TM_CCOEFF_NORMED)
                    result2 = cv2.matchTemplate(frame[y:y+h, x:x+w], resized_target_image, cv2.TM_CCOEFF_NORMED)

                    if result1.size == 0 or result2.size == 0:
                        print(f"MatchTemplate failed: result1 or result2 is empty.")
                        continue

                    average_vals.append(max(np.mean(result1), np.mean(result2)))

                print('Images compared')
                #print(average_vals)

                if average_vals:  # Make sure average_vals is not empty before dividing
                    if max(average_vals) >= 0:
                        prob += max(average_vals)
                    print('Final similarity probability:', prob)
                else:
                    print('No valid comparison results, skipping probability update.')

                if prob >= threshold:
                    return True, (x, y, w, h)
        v += 1
        if v >= len(vehicles):
            run = False
            return False, None
    
    return False, None

def detect_all(forground, car_crop, x, y, w, h, frame, frameCopy, color):
    #car_crop = frame[y:y + h, x:x + w]    #use frame or forground depending on video ---------------------------------
    #cv2.imshow('car_crop', car_crop)     #colored car with black bg
    
    dominant_color, _ = get_dominant_color(car_crop)       #works well on bright obvious colors (yellow, white, red)
    color_name_detected = color_name(dominant_color) 
    #time.sleep(1)   #for testing purposes
    
    license_plate = detect_license_plate(car_crop)      #very slow for some reason. likely not detected tho because difficult to get license plate unless very close/clear image
    
    if license_plate:        #change it to when the camera detects the car jacking, then it will store car info
        # Store car info
        store_car_info(color_name_detected, license_plate[0], [frame[y:y + h, x:x + w], car_crop], first_camera_starting_coords, [])   #(37.7749, -122.4194) woudl eb repalced by the first camera's actual coordiantes
    
    cv2.rectangle(frameCopy, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frameCopy, f"{color_name_detected}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    if license_plate:
        cv2.putText(frameCopy, f"{license_plate[0]}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return color_name_detected, license_plate

def after_detecting_specific_car(found, bbox, car_positions, frameCopy, directions, starting_coords, final_directions):
    if found:  
        x, y, w, h = bbox
        car_center = (x + w // 2, y + h // 2)
        car_positions.append(car_center)

        cv2.rectangle(frameCopy, (x, y), (x+w, y+h), (0, 0, 0), 3)
        cv2.putText(frameCopy, "User's Car Found", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if len(car_positions) > 1:    # already has positions in it
            delta_x = car_center[0] - car_positions[-2][0]
            delta_y = car_center[1] - car_positions[-2][1]

            if abs(delta_x) > abs(delta_y):  # Movement is more horizontal
                if delta_x > 0:
                    direction = "right"
                elif delta_x < 0:
                    direction = "left"
            else:  # Movement is more vertical
                #if delta_y > 0:   #ignore right now, because we position camera to make it so that car is always going forward/away from it
                    #direction = "down"   #ignore right now
                #elif delta_y < 0:      #ignore right now
                direction = "up"
            directions.append(direction)

    if len(car_positions) > 1 and len(directions) >= 1:
        last_position = car_positions[-1]
        margin_distance = 100
        if (last_position[0] < margin_distance and directions[-1] == 'left'):  # close to Left edge
            final_directions.append('left')
            print("Car went left.")
            car_positions, directions = call_user(car_positions, directions, starting_coords, final_directions)
            return car_positions, frameCopy, directions, final_directions, True
        elif (last_position[0] > (frame_width - margin_distance) and directions[-1] == 'right'):  # close to Right edge 
            final_directions.append('right')
            print("Car went right.")
            car_positions, directions = call_user(car_positions, directions, starting_coords, final_directions)
            return car_positions, frameCopy, directions, final_directions, True
        elif (last_position[1] < margin_distance and directions[-1] == 'up'):  # close to  Top edge
            final_directions.append('up')
            print("Car went straight (up).")
            car_positions, directions = call_user(car_positions, directions, starting_coords, final_directions)
            return car_positions, frameCopy, directions, final_directions, True
        elif (last_position[1] > (frame_height - margin_distance) and directions[-1] == 'down'):  # close to  Bottom edge
            final_directions.append('down')
            print("Car went straight (down).")
            car_positions, directions = call_user(car_positions, directions, starting_coords, final_directions)
            return car_positions, frameCopy, directions, final_directions, True
            
            # You could also check the sequence of angles (need to add some code earlier then using deltax and delta y)
            # to determine if the car made a U-turn or other complex maneuvers
    return car_positions, frameCopy, directions, final_directions, False

def call_user(car_positions, directions, starting_coords, final_directions):
    # Optional: Reset tracking if car exits frame
    car_positions.clear()
    directions.clear()

    user_response = input("Calling/messaging user about car's disappearance")   
    # Would have to be changed if used IRL, because we want to time user_resposne so we can check if they are not typing
    # anything. We would likely use threads or something here instead, but for simplicity, I used normal input field.

    if not user_response:    #if user doesn't respond (ex:sleeping) within 5 mins or if user responds and says that it's a carjacking, then 
        #from IRL_follow import follow_car   #normally this is what would be used IRL, but due to current camera limitations, we need use fake_follow.py instead
        #follow_car(starting_coords, final_directions)

        from fake_follow import main_follow
        main_follow(('J', 10), ['right', 'down', 'down', 'right', 'up'])    #it will sue the default exampel values. delete later
    
    return car_positions, directions

backgroundObject = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100, detectShadows=True)
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = None

stored_cars = load_car_info(folder='find_this')     #all stolen cars
if first_camera:
    user_car_path = 'user_car_images'        #for testing purposes, use user_car_images with cars.mp4, use user_car_images1 with yes_car.mp4 and no_car.mp4
    user_car_images = []
    for image_name in os.listdir(user_car_path):
        if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".PNG"):
            image_path = os.path.join(user_car_path, image_name)
            image = cv2.imread(image_path)
            user_car_images.append(image)

# Fixed reference point (center of the frame)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_point = (frame_width // 2, frame_height // 2)

directions = []
car_positions = []

while cap.isOpened():
    time.sleep(0.01)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    bodies = body_classifier.detectMultiScale(gray, 1.4, 2)
    buses = bus_classifier.detectMultiScale(gray, 1.4, 2)
    bikes = bike_classifier.detectMultiScale(gray, 1.4, 2)

    frameCopy = frame.copy()

    #isolate the cars on a black bg
    fgmask = backgroundObject.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)    #remove shadows
    fgmask = cv2.erode(fgmask, kernel1, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6)
    forground = cv2.bitwise_and(frame, frame, mask=fgmask)
    #cv2.imshow('forground', forground)     #colored cars with black bg
    #contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #for cnt in contours:
        #if cv2.contourArea(cnt) > 20000:
            #x, y, w, h = cv2.boundingRect(cnt)
            #detect_all(forground, x, y, w, h, frame, frameCopy, (255, 0, 0))   #green

    # Extract bounding boxes for any cars/bodies identified
    #for (x, y, w, h) in cars:    #use forgorund and contours isntead maybe ----------------------------------------
        #detect_all(forground, x, y, w, h, frame, frameCopy, (255, 0, 0))   #green
    #for (x,y,w,h) in bodies:
        #detect_all(forground, x, y, w, h, frame, frameCopy, (0, 0, 255))    #red
    #for (x,y,w,h) in buses:
        #detect_all(forground, x, y, w, h, frame, frameCopy, (0, 255, 0))  #blue
    #for (x,y,w,h) in bikes:
        #detect_all(forground, x, y, w, h, frame, frameCopy, (0, 255, 255))   #yellow

    # Iterate over stored cars and check for matches
    if first_camera:
        found, bbox = detect_specific_car(frame, fgmask, forground, user_car_color, user_car_license_plate, user_car_images)
        car_positions, frameCopy, directions, final_directions, a_break = after_detecting_specific_car(found, bbox, car_positions, frameCopy, directions, first_camera_starting_coords, final_directions)
    else:
        for car_info in stored_cars:        
            target_color = car_info["color"]
            target_license_plate = car_info["license_plate"]
            target_images = car_info["images"]
            starting_coords = car_info["starting_coords"]
            final_directions = car_info["final_directions"]

            # Basically the same thing as the if part
            found, bbox = detect_specific_car(frame, fgmask, forground, target_color, target_license_plate, target_images)
            car_positions, frameCopy, directions, final_directions, a_break = after_detecting_specific_car(found, bbox, car_positions, frameCopy, directions, starting_coords, final_directions)

    if a_break:
        break
    
    cv2.imshow('Detecting...', frameCopy)

    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frameCopy)
    frame_count += 1

    if cv2.waitKey(1) == ord('q'): 
        break

print('DONE')
cap.release()
cv2.destroyAllWindows()

frame_folder = 'frames'
frames = [f for f in sorted(os.listdir(frame_folder)) if f.endswith('.jpg')]

frame_path = os.path.join(frame_folder, frames[0])
frame = cv2.imread(frame_path)
height, width, layers = frame.shape

output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

for frame in frames:
    frame_path = os.path.join(frame_folder, frame)
    img = cv2.imread(frame_path)
    output_video.write(img)

output_video.release()
print('Video made successfully!')

#Limitations:
#0. It's not realistic to ahve a camera installed at every road, so maybe jsut neighborhoods probably and liek commercial stores
#1. Doesn't detect cars that are not in the frame and will always be delayed compared to the criminal
#2. Doesn't detect black cars and has trouble figuring out the color
#3. Camera must be in sky (maybe oblique) view of the road like in the cars.mp4 video fro findign directions easily. All cameras must have the exact same setup.
#5. The detection right now is pretty bad. Needs a lot of improvement.