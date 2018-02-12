% WORKING EXTRACTION OF ROTATION MATRICES
db = H36MDataBase.instance();
SUBJECT_IDS = [1 5 6 7 8 9 11]
cameras = [1 2 3 4]
cam_info = ones(3, 3, 0)


for s = SUBJECT_IDS
    for c = cameras
        cam = getCamera(db, s, c)
        cam_info = cat(3, cam_info, cam.R)
    end
end
save("rotationmatrices.mat")
% dlmwrite("rotationmatrices", cam_info)



% Working extraction but in dimensions that make sense!
db = H36MDataBase.instance();
SUBJECT_IDS = [1 5 6 7 8 9 11]
cameras = [1 2 3 4]
cam_info = []


for s = SUBJECT_IDS
    cam_info_over_camera = ones(3, 3, 0)
    for c = cameras
        cam = getCamera(db, s, c);
        cam_info_over_camera = cat(3, cam_info_over_camera, cam.R)
    end
    cam_info = cat(4, cam_info, cam_info_over_camera)
end
save("rotationmatrices.mat", "cam_info")



% Get the camera names as well
cam_names = []
for c = cameras
    cam = getCamera(db, 1, c)
    cam_names = [cam_names; cam.Name]
end
save("cameranames.mat", "cam_names")
