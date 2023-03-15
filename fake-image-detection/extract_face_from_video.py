# import libraries
import os
import cv2
import face_recognition

source = os.path.join(os.path.join(os.path.expanduser('~')), 'cnn/original data/faceforensics++/manipulated_sequences/NeuralTextures/c23/videos')


item_count = 0
for item in os.listdir(source):
  item_count += 1
  path = source+'/'+item
  if os.path.isfile(path):
    file = open(path)
    fileName,fileExtension = os.path.splitext(item)
    #print('It is a file whose extension is ' + fileExtension + ' and full name is ' + fileName + '. Size is ' + str(os.path.getsize(path)/1000))
    if fileExtension.endswith(('.mp4')):

      # Get a reference to webcam 
      # video_capture = cv2.VideoCapture("/dev/video1")
      video_capture = cv2.VideoCapture(path)

      # Initialize variables
      face_locations = []
      currentframe = 0
      while True:
          # Grab a single frame of video
          ret, frame = video_capture.read()
          if ret:
            # print(currentframe, item)
            currentframe += 1

            if currentframe % 25 == 0 or currentframe == 1:
              # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
              rgb_frame = frame[:, :, ::-1]

              # Find all the faces in the current frame of video
              face_locations = face_recognition.face_locations(rgb_frame)

              # Display the results
              for top, right, bottom, left in face_locations:
                  # Draw a box around the face
                  # a = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                  # crop_img = frame[top:bottom, left:right]
                  try:
                    crop_img = frame[top-15:bottom+15, left-15:right+15]
                    # cv2.imshow('Video', crop_img)
                    cv2.imshow("neuraltextures", crop_img)

                    name = './neuraltextures/' + fileName + '-frame_' + str(currentframe) + '.jpg'
                    print ('Creating...' + name, 'item: ', item_count)
          
                    # writing the extracted images
                    cv2.imwrite(name, crop_img)
                  except Exception as e:
                    print('>an error occurred')

              # Display the resulting image
              # cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # break
          else:
            break

      # Release handle to the webcam
      video_capture.release()
      cv2.destroyAllWindows()