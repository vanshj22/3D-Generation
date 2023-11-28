# combine point clouds


import open3d as o3d

intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
print(intrinsic.intrinsic_matrix)

x= o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
print(x)
# print(x.intrinsic_matrix)
o3d.io.write_pinhole_camera_intrinsic("test.json", x)
y= o3d.io.read_pinhole_camera_intrinsic("test.json")
print(y)

print("Read trajectory and cobimne")
pcds=[]
trajectory = o3d.io.read_pinhole_camera_trajectory("RGBD/normal_map.npy")
o3d.io.write_pinhole_camera_trajectory("test.json", trajectory) 

print("Trajectory Extrinsics:")
print(trajectory.parameters[0].extrinsic)  

for i in range(5):
    im1 = o3d.io.read_image("livingroom1_clean_micro/depth/{:05d}.png".format(i))
    im2 = o3d.io.read_image("livingroom1_clean_micro/image/{:05d}.png".format(i))  
    im = o3d.geometry.RGBDImage.create_from_color_and_depth(im1, im2,1000, 0, 5.0, False)
    
    intrinsic = trajectory.parameters[i].intrinsic
    extrinisic = trajectory.parameters[i].extrinsic
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im1, im2, intrinsic, extrinisic)
    pcds.append(pcd)

o3d.visualization.draw_geometries(pcds)




