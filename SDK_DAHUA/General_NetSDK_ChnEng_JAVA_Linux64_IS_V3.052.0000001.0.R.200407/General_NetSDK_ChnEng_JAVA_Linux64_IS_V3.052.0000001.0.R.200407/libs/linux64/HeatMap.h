#ifndef _HEATMAP_H__
#define _HEATMAP_H__

#if (defined(WIN32) || defined(_WIN32) || defined(_WIN64))

	#ifdef HEATMAP_EXPORTS
	#define HEATMAP_API  __declspec(dllexport)
	#else
	#define HEATMAP_API  __declspec(dllimport)
	#endif

	#define CALLBACK        __stdcall
	#define CALLMETHOD     __stdcall  //__cdecl
#else    //non-windows
	#define HEATMAP_API  extern "C"
	#define CALLMETHOD 

#endif


#ifdef	__cplusplus
extern "C" {
#endif

	//Bmp位图信息
	typedef struct bmpImageInfo
	{
		unsigned char *pBuffer;			//Bmp图片数据指针
		int nWidth;						//图片宽度
		int nHeight;					//图片高度
		int nBitCount;					//图片位数,支持8位，24位，32位
		int nDirection;                 //数据存储方向 0：从上到下，从左到右， 1：从下到上，从左到右
	}BMPIMAGE_INFO;

	///输入数据信息
	typedef struct heatMapInfoIn
	{
		BMPIMAGE_INFO stuGrayBmpInfo;		//8位Bmp灰度热度图数据：不包含图片头，数据存储方向从上到下
		BMPIMAGE_INFO stuBkBmpInfo;			//背景图Bmp位图数据：包含图片头，存储方向从下到上
	}HEATMAP_IMAGE_IN;

	//输出数据信息
	typedef struct heatMapInfoOut
	{
		unsigned char *pBuffer;				 //输出的彩色热度图数据（包含图片头）,宽高、位数和背景图相同
		int		nPicSize;					 //图片内存大小(包含头) ：宽*高*nBitCount/8 + 54
		float  fOpacity;					 //透明度,范围0-1
	}HEATMAP_IMAGE_Out;

	///\brief 生成热度图数据信息
	/// param [in] stuBmpInfoIn       Bmp位图数据输入
	/// param [in] stuBmpInfoOut      Bmp位图数据输出,包含图片头
	/// param [out] true or false
	HEATMAP_API bool CALLMETHOD CreateHeatMap(const HEATMAP_IMAGE_IN *stuBmpInfoIn, HEATMAP_IMAGE_Out *stuBmpInfoOut);

#ifdef __cplusplus
}
#endif


#endif  //_HEATMAP_H__