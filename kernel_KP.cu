#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>

#include <ctime>
#include <iostream>

#pragma comment(lib, "glew32.lib")
void mouseWheel(int wheel, int direction, int x, int y);
void mouse(int button, int state, int x, int y);

const int w = 1024;
const int h = 648;
const float a1 = 0.01;
const float a2 = 0.5;
const float inertia = 0.8;

struct my_float3{
	float x, y, z;
};

FILE* output;
const int birds_cnt = 5000;

float2 * birds_coord;
float2* birds_speed;

float2 * dev_birds_coord;
float2* dev_birds_speed;
float2* dev_birds_force;

thrust::device_ptr<my_float3> dev_ptr_min;
my_float3* dev_min_vec;

dim3 blocks(32, 32), threads(16, 16);
int blocks1 = 1024, threads1 = 1024;
float* dev_rand_numbers;

#define CSC(call) {							\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));		\
		system("pause");                \
        exit(1);							\
	    }									\
} while (0)

#define sqr(x) ((x)*(x))


float xc = 0.0f, yc = 0.0f, sx = 0.3f, sy = sx * h / w, minf = -3.0, maxf = 3.0, p = 0.0;

__constant__ float dev_xc, dev_yc, dev_sx, dev_sy, dev_minf, dev_maxf, dev_p;

__device__ float fun(float x, float y) {
	
	return 100 * (y - x*x)*(y - x*x) + (x - 1)*(x - 1); 
}

__device__ float fun(int i, int j)  {
	float x = 2.0f * i / (float)(w - 1) - 1.0f;
	float y = 2.0f * j / (float)(h - 1) - 1.0f;	
	return fun(x * dev_sx + dev_xc, -y * dev_sy + dev_yc);	 
}

__device__ float rand01(float* dev_rand_numbers, int n)
{	
	return dev_rand_numbers[threadIdx.x + n*blockIdx.x * blockDim.x];
}



__device__ int2 coordToIndex(float2 x){             
	int i = (int)(((x.x - dev_xc) / dev_sx + 1)*((w - 1) / 2.));
	int j = (int)(((x.y - dev_yc) / (-dev_sy) + 1)*((h - 1) / 2.));
	return make_int2(i, j);
}

__device__ float2 indexToCoord(int i, int j)  {   
	return make_float2(
		(2.0f * i / (float)(w - 1) - 1.0f) * dev_sx + dev_xc,
		-(2.0f * j / (float)(h - 1) - 1.0f) * dev_sy + dev_yc
		);
}

__host__ float2 indexToCoordHost(int i, int j)  {
	return make_float2(
		(2.0f * i / (float)(w - 1) - 1.0f) * sx + xc,
		-(2.0f * j / (float)(h - 1) - 1.0f) * sy + yc
		);
}

 int2 coordToIndexHost(float2 x){
	int i = (int)(((x.x - xc) / sx + 1)*((w - 1)/2.));
	int j = (int)(((x.y - yc) / (-sy) + 1)*((h - 1)/2.));
	return make_int2(i, j);
}


__global__ void cudaRand(float *d_out, int size, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);
    
    d_out[i + size*n] = curand_uniform(&state);
}

__global__ void heightMap(uchar4* data) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	for(i = idx; i < w; i += offsetx)
		for(j = idy; j < h; j += offsety) {
			float f = (fun(i, j) - dev_minf) / (dev_maxf - dev_minf); 
			//data[j * w + i] = make_uchar4(0, 0, (int)(f * 255), 255);
			data[j * w + i] = make_uchar4((int)(f * 200 ), 0, (int)((1-f) * 200), 255);
			//data[j * w + i] = make_uchar4(0, 0, 0, 255);
		}
}

__global__ void setBirds(uchar4* data, float2* birds_coord, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += offsetx) 
	{
	 	int2 index = coordToIndex(birds_coord[i]);
		if (index.x >= 0 && index.x < w && index.y >=0 && index.y < h)
		{
			data[index.y * w + index.x] = make_uchar4(255, 255, 255, 255);

			
			if (index.x+1 >= 0 && index.x+1 < w && index.y >=0 && index.y < h)
				data[index.y * w + index.x+1] = make_uchar4(255, 255, 255, 255);
			if (index.x-1 >= 0 && index.x-1 < w && index.y >=0 && index.y < h)
				data[index.y * w + index.x-1] = make_uchar4(255, 255, 255, 255);
			if (index.x >= 0 && index.x < w && index.y-1 >=0 && index.y -1 < h)
				data[(index.y-1) * w + index.x] = make_uchar4(255, 255, 255, 255);
			if (index.x >= 0 && index.x < w && index.y + 1 >= 0 && index.y + 1 < h)
				data[(index.y+1) * w + index.x] = make_uchar4(255, 255, 255, 255);
		}

	}

}

__global__ void getBirdsMin(float2* birds_coord, thrust::device_ptr<my_float3> dev_ptr_min, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += offsetx) 
	{
		
		int2 param = coordToIndex(birds_coord[i]);
		float cur_val = fun(param.x, param.y);
		

		my_float3 min = dev_ptr_min[i];
		float min_val = min.z;
		if (cur_val < min_val)
		{
			min.x = birds_coord[i].x;
			min.y = birds_coord[i].y;
			min.z = cur_val;
			dev_ptr_min[i] = min;
		}
	}
}

__global__ void correctSpeedAndCoord(float2* birds_coord, thrust::device_ptr<my_float3> ptr_min, float2* birds_speed, float3 global_min, 
	const float inertia, const float a1, const float a2, int size, float* rand, float2* force)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += offsetx)
	{
		my_float3 pbest = ptr_min[i];
		birds_speed[i].x = inertia*birds_speed[i].x + a1*rand01(rand, 0)*(pbest.x - birds_coord[i].x) + a2*rand01(rand, 1)*(global_min.x - birds_coord[i].x) + force[i].x*(-0.01);
		birds_speed[i].y = inertia*birds_speed[i].y + a1*rand01(rand, 0)*(pbest.y - birds_coord[i].y) + a2*rand01(rand, 1)*(global_min.y - birds_coord[i].y) + force[i].y*(-0.01);
		birds_coord[i].x += birds_speed[i].x;
		birds_coord[i].y += birds_speed[i].y;		
	}
}

__global__ void getForce(float2* birds_coord, float2* birds_force, int size)
{
	const float EPS = 1e-4;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < size; i += offsetx)
	{
		birds_force[i] = make_float2(0, 0);
		for (int j = 0; j < size; j++)
		{
			if (j == i)
				continue;
			float dist = sqr(sqr(birds_coord[i].x - birds_coord[j].x) + sqr(birds_coord[i].y - birds_coord[j].y));
			birds_force[i].x += (birds_coord[i].x - birds_coord[j].x) / (dist + EPS);
			birds_force[i].y += (birds_coord[i].y - birds_coord[j].y) / (dist + EPS);
		}
	}
}

struct cudaGraphicsResource *res;

struct min_my_float3 {
	__device__ my_float3 operator()(const my_float3& a, const my_float3& b) const {
		if (a.z < b.z) return a; 
		else return b;
	}
};

struct add_float2 {
	__device__ float2 operator()(const float2& a, const float2& b) const {
		float2 r;
		r.x = a.x + b.x;
		r.y = a.y + b.y;
		return r;
	}
};

void update() {
	uchar4* dev_data;
	size_t size;
	p += 0.05f; 
	
	CSC(cudaMemcpyToSymbol((const void*)&dev_sx, &sx, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_p, &p, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_sy, &sy, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_xc, &xc, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_yc, &yc, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_minf, &minf, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_maxf, &maxf, sizeof(float)));

	CSC(cudaMemcpyToSymbol((const void*)&dev_p, &p, sizeof(float)));
	CSC(cudaGraphicsMapResources(1, &res, 0)); 
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));
	heightMap<<<blocks, threads>>>(dev_data);
	setBirds<<<blocks1, threads1>>>(dev_data, dev_birds_coord, birds_cnt);
	
	
	
	getBirdsMin<<<blocks1, threads1>>>(dev_birds_coord, dev_ptr_min, birds_cnt);

	
	getForce << <blocks1, threads1 >> >(dev_birds_coord, dev_birds_force, birds_cnt);

	
	my_float3 global_min = thrust::reduce(dev_ptr_min, dev_ptr_min + birds_cnt - 1, my_float3{ 0, 0, 1e10 }, min_my_float3());
	
	int2 cti = coordToIndexHost(make_float2(global_min.x, global_min.y));
	//printf("%d %d\n", cti.x, cti.y);
	float3 global_min_param = make_float3(global_min.x, global_min.y, global_min.z);
	for (int i = 0; i < 2; i++)
		cudaRand << <blocks1, threads1 >> >(dev_rand_numbers, blocks1*threads1, i);
	correctSpeedAndCoord << <blocks1, threads1 >> >(dev_birds_coord, dev_ptr_min, dev_birds_speed, global_min_param, 
		inertia, a1, a2, birds_cnt, dev_rand_numbers, dev_birds_force);


	thrust::device_ptr<float2> new_coord_ptr(dev_birds_coord);
	float2 new_coord = thrust::reduce(new_coord_ptr, new_coord_ptr + birds_cnt - 1, make_float2(0,0), add_float2());
	new_coord.x /= birds_cnt;
	new_coord.y /= birds_cnt;
	xc = new_coord.x;
	yc = new_coord.y;
	CSC(cudaDeviceSynchronize());


	CSC(cudaGraphicsUnmapResources(1, &res, 0));

	glutPostRedisplay(); 
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0); 
	glClear(GL_COLOR_BUFFER_BIT); 
	glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0); 
	glutSwapBuffers(); 
}


int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE| GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("Hot map"); 

	glutIdleFunc(update); 
	glutDisplayFunc(display); 

	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity(); 
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h); 

	glewInit();
	
	CSC(cudaMemcpyToSymbol((const void*)&dev_sx, &sx, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_p, &p, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_sy, &sy, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_xc, &xc, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_yc, &yc, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_minf, &minf, sizeof(float)));
	CSC(cudaMemcpyToSymbol((const void*)&dev_maxf, &maxf, sizeof(float)));

	birds_coord = (float2*)malloc(sizeof(float2)*birds_cnt);
	birds_speed = (float2*)malloc(sizeof(float2)*birds_cnt);
	for (int i = 0; i < birds_cnt; i++)
	{		
		birds_coord[i].x = rand() % 101 * 0.01 * (rand() % 2 == 0 ? -1 : 1) * sx + xc;
		birds_coord[i].y = rand() % 101 * 0.01 * (rand() % 2 == 0 ? -1 : 1) * (-1) * sy + yc;
		birds_speed[i].x = rand() % 10*0.01;											
		birds_speed[i].y = rand() % 10*0.01;
	}
	CSC(cudaMalloc((void**)&dev_birds_coord, birds_cnt* sizeof(float2)));
	CSC(cudaMemcpy(dev_birds_coord, birds_coord, birds_cnt* sizeof(float2), cudaMemcpyHostToDevice));
	CSC(cudaMalloc((void**)&dev_birds_speed, birds_cnt* sizeof(float2)));
	CSC(cudaMemcpy(dev_birds_speed, birds_speed, birds_cnt* sizeof(float2), cudaMemcpyHostToDevice));


	thrust::device_vector<my_float3> min_vec(birds_cnt); 
	for (int i = 0; i < birds_cnt; i++)
	{
		min_vec[i] = my_float3{ birds_coord[i].x, birds_coord[i].y, 1e10 };
	}
	dev_ptr_min = thrust::device_pointer_cast(min_vec.data());

	output = fopen("res.txt", "w");
	
	CSC(cudaMalloc((void**)&dev_rand_numbers, 2*blocks1*threads1*sizeof(float)));
	
	CSC(cudaMalloc((void**)&dev_birds_force, birds_cnt* sizeof(float2)));
	


	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
	
	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));


	glutMouseWheelFunc(mouseWheel);
	glutMouseFunc(mouse);

	glutMainLoop(); 
	
	CSC(cudaGraphicsUnregisterResource(res));

	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	CSC(cudaFree(dev_rand_numbers));
	CSC(cudaFree(dev_birds_speed));
	CSC(cudaFree(dev_birds_coord));
	CSC(cudaFree(dev_birds_force));
	min_vec.clear();
	free(birds_coord);
	free(birds_speed);
	
	return 0;
}



void mouseWheel(int wheel, int direction, int x, int y)
{
	if (direction > 0){
		sx -= 0.05f;
		sx <= 0 ? sx = 0.01f : sx;
		sy = sx * h / w;
	}
	if (direction < 0){
		sx += 0.05f;
		sx > 1000 ? sx = 1000.0f : sx;
		sy = sx * h / w;
	}
}


void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON){
		if (state == GLUT_DOWN){
			float2 new_coord = indexToCoordHost(x, y);
			xc = new_coord.x;
			yc = new_coord.y;
		}
	}
}