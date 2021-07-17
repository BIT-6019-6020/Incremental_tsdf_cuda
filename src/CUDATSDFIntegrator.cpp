//
// Created by will on 20-1-9.
//

#include "CUDATSDFIntegrator.h"
#include "parameters.h"
#include <numeric>

void
IntegrateDepthMapCUDA(float *d_cam_K,
                      float *T_bc,
                      float *d_depth,
                      uchar3 *d_color,
                      float voxel_size,
                      float truncation,
                      int height,
                      int width,
                      int grid_dim_x,
                      int grid_dim_y,
                      int grid_dim_z,
                      float gird_origin_x,
                      float gird_origin_y,
                      float gird_origin_z,
                      Voxel *d_SDFBlocks,
                      float *d_exceed_num);

void
deIntegrateDepthMapCUDA();

CUDATSDFIntegrator::CUDATSDFIntegrator()
{
    // Camera Intrinsics
    h_camK[0] = (float) Cfgparam.fx;
    h_camK[1] = (float) Cfgparam.fy;
    h_camK[2] = (float) Cfgparam.cx;
    h_camK[3] = (float) Cfgparam.cy;

    std::cout << "[fx,fy,cx,cy]: " << h_camK[0] << "," << h_camK[1] << "," << h_camK[2] << "," << h_camK[3]
              << std::endl;

    h_width = Cfgparam.img_size.width;
    h_height = Cfgparam.img_size.height;
    h_voxelSize = Cfgparam.VoxelSize;
    h_truncation = Cfgparam.Truncation;

    h_gridSize_x = Cfgparam.GridSize_x;
    h_gridSize_y = Cfgparam.GridSize_y;
    h_gridSize_z = Cfgparam.GridSize_z;
}

void
CUDATSDFIntegrator::Initialize(cv::Mat &depth_img, cv::Mat &depth_gt_img, Eigen::Matrix4d &e_Twc,
                               float *depth, double scale_gt_div_est)
{
    int width = Cfgparam.img_size.width;
    int height = Cfgparam.img_size.height;
    memset(depth, 0.0f, width * height); //清零
    int valid_cnter = 0;
    int start_line = 0;

    if (!Cfgparam.gt1_or_est0 && Cfgparam.apply_scale)
    {
        for (int y = 0; y < height; y++)
        {
            int valid_pixel_cnter = 0;
            for (int x = 0; x < width; x++)
            {
                double est_val = (double) depth_img.at<char16_t>(y, x) / Cfgparam.depth_result_factor;

                if (est_val <= 0.0 || isnan(est_val) || isnanf(est_val))
                    continue;
                valid_pixel_cnter++;
            }

            if (valid_pixel_cnter > width * 0.6)
            {
                valid_cnter++;
            }
            else
            {
                valid_cnter = 0;
            }

            if (valid_cnter >= Cfgparam.n_line2giveup)
            {
                start_line = y;
                break;
            }
        }
    }

    vector<float> pt_x;
    vector<float> pt_y;
    vector<float> pt_z;

    for (int r = start_line; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            float curr_depth;
            if (Cfgparam.gt1_or_est0 && Cfgparam.apply_scale)
            {
                curr_depth =
                    (float) ((depth_gt_img.at<char16_t>(r, c) / Cfgparam.depth_gt_mfactor) + Cfgparam.depth_gt_dfactor);
                curr_depth /= (float) scale_gt_div_est;
            }
            else
            {
                curr_depth = (float) ((depth_img.at<char16_t>(r, c)) / Cfgparam.depth_result_factor);
            }


            if (curr_depth <= 0 || isinf(curr_depth) || isnan(curr_depth))
                depth[r * width + c] = 0.0;
            else
            {
                float pt_pix_x = ((float) c - (float) Cfgparam.cx) / (float) Cfgparam.fx;
                float pt_pix_y = ((float) r - (float) Cfgparam.cy) / (float) Cfgparam.fy;
                Eigen::Vector4f pt_b(pt_pix_x * curr_depth, pt_pix_y * curr_depth, curr_depth, 1.0f);

                pt_x.push_back(pt_b.x());
                pt_y.push_back(pt_b.y());
                pt_z.push_back(pt_b.z());
            }
            depth[r * width + c] = curr_depth;
            if (depth[r * width + c] > 6.0 || depth[r * width + c] < 0.2)
                depth[r * width + c] = 0.0; // Only consider depth < 6m
        }
    }

    float mean_x;
    float mean_y;
    float mean_z;

    float stdv_x;
    float stdv_y;
    float stdv_z;

    vectorSumMean(pt_x, mean_x, stdv_x);
    vectorSumMean(pt_y, mean_y, stdv_y);
    vectorSumMean(pt_z, mean_z, stdv_z);

    int sum_test_x = 0;
    int sum_test_y = 0;
    int sum_test_z = 0;
    float factor = Cfgparam.factor_voxel;


    h_grid_origin_x = (float) mean_x - (h_voxelSize * h_gridSize_x) / 2.0;
    h_grid_origin_y = ((float) mean_y + factor * stdv_y) - (h_gridSize_y * h_voxelSize);
    h_grid_origin_z = (float) mean_z - (h_voxelSize * h_gridSize_z) / 2.0;

    h_grid_end_x = h_grid_origin_x + h_voxelSize * h_gridSize_x;
    h_grid_end_y = h_grid_origin_y + h_voxelSize * h_gridSize_y;
    h_grid_end_z = h_grid_origin_z + h_voxelSize * h_gridSize_z;

    for (int i = 0; i < pt_z.size(); ++i)
    {
        if ((pt_z[i] >= h_grid_origin_z) &&
            (pt_z[i] <= h_grid_end_z)
            )
        {
            sum_test_z++;
        }

        if ((pt_y[i] >= h_grid_origin_y) &&
            (pt_y[i] <= h_grid_end_y)
            )
        {
            sum_test_y++;
        }

        if ((pt_x[i] >= h_grid_origin_x) &&
            (pt_x[i] <= h_grid_end_x)
            )
        {
            sum_test_x++;
        }
    }


    std::cout << "  x percent: " << (double) sum_test_x / (double) pt_x.size() << std::endl;
    std::cout << "  x mean: " << mean_x << " stdv_z:" << stdv_z << std::endl;

    std::cout << "  y percent: " << (double) sum_test_y / (double) pt_y.size() << std::endl;
    std::cout << "  z percent: " << (double) sum_test_z / (double) pt_z.size() << std::endl;

    std::cout << "  x range: [" << h_grid_origin_x << "--" << h_grid_end_x << "] " << std::endl;
    std::cout << "  y range: [" << h_grid_origin_y << "--" << h_grid_end_y << "] " << std::endl;
    std::cout << "  z range: [" << h_grid_origin_z << "--" << h_grid_end_z << "] " << std::endl;

    std::cout << "Truncation: " << h_truncation << std::endl;
    std::cout << " VoxelSize: " << h_voxelSize << std::endl;
    std::cout << "  GridSize: " << h_gridSize_x << std::endl;
    std::cout << "Initialize TSDF ..." << std::endl;

    if (!is_initialized)
    {
        h_SDFBlocks = new Voxel[h_gridSize_x * h_gridSize_y * h_gridSize_z];

        checkCudaErrors(cudaMalloc(&d_camK, 4 * sizeof(float)));

        checkCudaErrors(cudaMemcpy(d_camK, h_camK, 4 * sizeof(float), cudaMemcpyHostToDevice));
        // TSDF model
        checkCudaErrors(cudaMalloc(&d_SDFBlocks, h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel)));
        // depth data
        checkCudaErrors(cudaMalloc(&d_depth, h_height * h_width * sizeof(float)));
        // color data
        checkCudaErrors(cudaMalloc(&d_color, h_height * h_width * sizeof(uchar3)));
        // pose in base coordinates
        checkCudaErrors(cudaMalloc(&T_bc, 4 * 4 * sizeof(float)));

        checkCudaErrors(cudaMalloc(&exceed_num, 3 * sizeof(float)));

        is_initialized = true;
    }

    if (need_reset)
    {
        delete[]h_SDFBlocks;
        h_SDFBlocks = new Voxel[h_gridSize_x * h_gridSize_y * h_gridSize_z];
        checkCudaErrors(cudaMemcpy(d_SDFBlocks, h_SDFBlocks,
                                   h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyHostToDevice));
        need_reset = false;
    }
}

Voxel *
CUDATSDFIntegrator::point(int pt_grid_x, int pt_grid_y, int pt_grid_z)
{

    int volume_idx = pt_grid_z * h_gridSize_x * h_gridSize_y +
        pt_grid_y * h_gridSize_x +
        pt_grid_x;
    return &h_SDFBlocks[volume_idx];
}

// Integrate depth and color into TSDF model
void
CUDATSDFIntegrator::integrate(float *depth_cpu_data, uchar3 *color_cpu_data, float *T_bc_)
{
    //std::cout << "Fusing color image and depth" << std::endl;

    // copy data to gpu
    TicToc timer;
    checkCudaErrors(cudaMemcpy(d_depth, depth_cpu_data, h_height * h_width * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_color, color_cpu_data, h_height * h_width * sizeof(uchar3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(T_bc, T_bc_, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice));


    cout << "load GPU memory cost:" << timer.toc() << " ms" << endl;

    // Integrate function
    IntegrateDepthMapCUDA(d_camK,
                          T_bc,
                          d_depth,
                          d_color,
                          h_voxelSize,
                          h_truncation,
                          h_height,
                          h_width,
                          h_gridSize_x,
                          h_gridSize_y,
                          h_gridSize_z,
                          h_grid_origin_x,
                          h_grid_origin_y,
                          h_grid_origin_z,
                          d_SDFBlocks,
                          exceed_num
    );

    checkCudaErrors(cudaMemcpy(h_exceed_num, exceed_num,
                               3 * sizeof(float), cudaMemcpyDeviceToHost));


    cout << "fusion cost:" << timer.toc() << " ms" << endl;
    FrameId++;
}

// deIntegrate depth and color from TSDF model
void
CUDATSDFIntegrator::deIntegrate(float *depth_cpu_data, uchar3 *color_cpu_data, float *pose)
{

}

// Compute surface points from TSDF voxel grid and save points to point cloud file
void
CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud(float tsdf_thresh, float weight_thresh, Eigen::Matrix4d Twc)
{

    checkCudaErrors(cudaMemcpy(h_SDFBlocks, d_SDFBlocks,
                               h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyDeviceToHost));

    pcl::PointCloud<pcl::PointXYZRGB> curr_pointcloud;
    curr_scene->clear();

    for (int i = 0; i < h_gridSize_x * h_gridSize_y * h_gridSize_z; i++)
    {
        if (std::abs(h_SDFBlocks[i].sdf) < tsdf_thresh && h_SDFBlocks[i].weight > weight_thresh)
        {
            // Compute voxel indices in int for higher positive number range
//            这里的xyz是指格子的排列坐标
            int z = floor(i / (h_gridSize_x * h_gridSize_y));
            int y = floor((i - (z * h_gridSize_x * h_gridSize_y)) / h_gridSize_x);
            int x = i - (z * h_gridSize_x * h_gridSize_y) - y * h_gridSize_x;

//            这里的pt_base xyz指的是在基坐标下的绝对坐标
            float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize;
            float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize;
            float pt_base_z = h_grid_origin_z + (float) z * h_voxelSize;

            pcl::PointXYZRGB point;
            point.x = pt_base_x;
            point.y = pt_base_y;
            point.z = pt_base_z;

            point.r = h_SDFBlocks[i].color.x;
            point.g = h_SDFBlocks[i].color.y;
            point.b = h_SDFBlocks[i].color.z;
            curr_scene->push_back(point);
        }
    }

    need_reset = true;
    pcl::transformPointCloud(*curr_scene, *curr_scene, Twc);
}

void
CUDATSDFIntegrator::SaveVoxelGrid2SurfacePointCloud_1(float tsdf_thresh, float weight_thresh, Eigen::Matrix4d Twc)
{

    checkCudaErrors(cudaMemcpy(h_SDFBlocks, d_SDFBlocks,
                               h_gridSize_x * h_gridSize_y * h_gridSize_z * sizeof(Voxel), cudaMemcpyDeviceToHost));

    pcl::PointCloud<pcl::PointXYZRGB> curr_pointcloud;
    curr_scene->clear();


    for (int y = 0; y < h_gridSize_y; ++y)
    {
        for (int x = 0; x < h_gridSize_x; ++x)
        {
            Voxel *best_voxel = nullptr;
            float sum_z = 0;
            float sum_weight = 0;
            int best_z = -1;
            for (int z = 0; z < h_gridSize_z; ++z)
            {
                Voxel *curr_voxel = point(x, y, z);

                if (std::abs(curr_voxel->sdf) < tsdf_thresh && curr_voxel->weight > weight_thresh)
                {
                    if (best_voxel == nullptr)
                    {
                        best_voxel = curr_voxel;
                        best_z = z;
                    }
                    else
                    {
                        sum_weight += curr_voxel->weight;
                        sum_z += curr_voxel->weight * (float) z;

                        if (curr_voxel->weight > best_voxel->weight && curr_voxel->sdf < best_voxel->sdf)
                        {
                            best_voxel = curr_voxel;
                            best_z = z;
                        }
                    }
                }
            }

            if (best_z != -1)
            {
                float best_z_ = sum_z / sum_weight;
                float pt_base_x = h_grid_origin_x + (float) x * h_voxelSize;
                float pt_base_y = h_grid_origin_y + (float) y * h_voxelSize;
                float pt_base_z = h_grid_origin_z + (float) best_z_ * h_voxelSize;

                pcl::PointXYZRGB point;
                point.x = pt_base_x;
                point.y = pt_base_y;
                point.z = pt_base_z;

                point.r = best_voxel->color.z;
                point.g = best_voxel->color.y;
                point.b = best_voxel->color.x;
                curr_scene->push_back(point);
            }

        }
    }

    need_reset = true;
    pcl::transformPointCloud(*curr_scene, *curr_scene, Twc);
}

// Default deconstructor
CUDATSDFIntegrator::~CUDATSDFIntegrator()
{
    free(h_SDFBlocks);
    checkCudaErrors(cudaFree(d_camK));
    checkCudaErrors(cudaFree(d_SDFBlocks));
    checkCudaErrors(cudaFree(d_depth));
    checkCudaErrors(cudaFree(d_color));
}
