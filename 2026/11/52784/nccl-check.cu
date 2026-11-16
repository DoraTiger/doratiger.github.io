#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define C(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));return 2;}}while(0)
#define N(x) do{ncclResult_t e=(x);if(e!=ncclSuccess){fprintf(stderr,"NCCL %s\n",ncclGetErrorString(e));return 3;}}while(0)
int main(int argc,char**argv){
 if(argc!=4)return 1; int rank=atoi(argv[1]),gpu=atoi(argv[3]); ncclUniqueId id;
 if(rank==0){N(ncclGetUniqueId(&id));FILE*f=fopen(argv[2],"wb");if(!f)return 4;fwrite(&id,sizeof(id),1,f);fclose(f);}
 else {
 FILE*f=fopen(argv[2],"rb");if(!f)return 4;if(fread(&id,sizeof(id),1,f)!=1)return 4;fclose(f);}
 C(cudaSetDevice(gpu));ncclComm_t comm;N(ncclCommInitRank(&comm,2,id,rank));
 size_t n=1<<20;std::vector<float> h(n,float(rank+1));float*d;C(cudaMalloc(&d,n*sizeof(float)));C(cudaMemcpy(d,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));
 N(ncclAllReduce(d,d,n,ncclFloat,ncclSum,comm,0));C(cudaDeviceSynchronize());C(cudaMemcpy(h.data(),d,n*sizeof(float),cudaMemcpyDeviceToHost));
 for(size_t i=0;i<n;i++)if(h[i]!=3.0f){fprintf(stderr,"MISMATCH %zu %f\n",i,h[i]);return 5;}
 printf("ALLREDUCE_PASS rank=%d gpu=%d elements=%zu\n",rank,gpu,n);N(ncclCommDestroy(comm));C(cudaFree(d));return 0;
}
