/*
+	LBF3000
+	A Implementation of highly efficient and very accurate regression approach for face alignment.
	Quantum Dynamics Co.,Ltd. 量子动力（深圳）计算机科技有限公司

	Based on the paper 'Face Alignment at 3000 FPS via Regressing Local Binary Features'
	University of Science and Technology of China.
	Microsoft Research.

+	'LBF3000' is developing under the terms of the GNU General Public License as published by the Free Software Foundation.
+	The project lunched by 'Quantum Dynamics Lab.' since 4.Aug.2017.

+	You can redistribute it and/or modify it under the terms of the GNU General Public License version 2 (GPLv2) of
+	the license as published by the free software foundation.this program is distributed in the hope
+	that it will be useful,but without any warranty.without even the implied warranty of merchantability
+	or fitness for a particular purpose.

+	This project allows for academic research only.
+	本项目代码仅授权于学术研究，不可用于商业化。

+	(C)	Quantum Dynamics Lab. 量子动力实验室
+		Website : http://www.facegood.cc
+		Contact Us : jelo@facegood.cc

+		-Thanks to Our Committers and Friends
+		-Best Wish to all who Contributed and Inspired
*/

/*
+	Start Train & Start Predict

+	Date:		2017/7/20
+	Author:		ZhaoHang
*/
#include <fstream>
#include <ppl.h>
#include <Mutex>
#include <direct.h>

#include "FgLBFTrain.h"

FgLBFTrain::FgLBFTrain(string TrainPath)
	: m_TrainPath(TrainPath)
{
}

FgLBFTrain::~FgLBFTrain()
{
}

void FgLBFTrain::Load()
{
	std::ifstream TrainConfig(m_TrainPath + "TrainConfig.txt");
	if (!TrainConfig.is_open())
		ThrowFaile;

	TrainConfig >> g_TrainParam.LocalFeaturesNum
		>> g_TrainParam.LandmarkNumPerFace
		>> g_TrainParam.RegressroStage
		>> g_TrainParam.TreeDepth
		>> g_TrainParam.TreeNumPerForest
		>> g_TrainParam.DataAugmentScale
		>> g_TrainParam.DataAugmentOverLap;

	g_TrainParam.LocalRadiusPerStageVec.resize(g_TrainParam.RegressroStage);
	for (int32_t i = 0; i < g_TrainParam.RegressroStage; ++i)
	{
		TrainConfig >> g_TrainParam.LocalRadiusPerStageVec[i];
	}

	string TrainImageList;
	TrainConfig >> TrainImageList;

	LoadImageList(m_TrainPath + TrainImageList, g_ImageVec, g_TruthShapeVec, g_BoxVec);

	int32_t HasTestImage;
	TrainConfig >> HasTestImage;
	if (HasTestImage > 0)
	{
		TrainConfig >> m_TestImagePath;
	}
	TrainConfig.close();

	g_TrainParam.MeanShape = GetMeanShape(g_TruthShapeVec, g_BoxVec);

	//Data Augment
	DataAugment();
}

void FgLBFTrain::Train()
{
	Load();
	m_RegressorVec.resize(g_TrainParam.RegressroStage, FgLBFRegressor(m_FaceDataVec));
	for (int32_t i = 0; i < m_RegressorVec.size(); ++i)
	{
		std::cout << "Train Stage " << i << std::endl;
		vector<Mat_d> Delta = m_RegressorVec[i].Train(i);
		for (int32_t j = 0; j < m_FaceDataVec.size(); ++j)
			m_FaceDataVec[j].CurrentShape += Delta[j];
	}

	SaveToPath(m_TrainPath + "Model/");

	g_ImageVec.clear();
	g_TruthShapeVec.clear();
	g_BoxVec.clear();

	if (!m_TestImagePath.empty())
	{
		std::cout << "Test Image" << std::endl;

		vector<Mat_uc>			ImageVec;
		vector<Mat_d>			TruthShapeVec;
		vector<cv::Rect>		BoxVec;
		LoadImageList(m_TrainPath + "Test_Image.txt", ImageVec, TruthShapeVec, BoxVec);

		double_t E = 0;
		for (int32_t i = 0; i < ImageVec.size(); ++i)
		{
			Mat_d PredictShape = Predict(ImageVec[i], BoxVec[i]);
			E += CalculateError(TruthShapeVec[i], PredictShape);
		}
		std::cout << "Error :[" << E << "]\t Mean Error :[" << E / ImageVec.size() << "]" << std::endl;
	}
}

void FgLBFTrain::Predict(string ImageListPath)
{
	LoadFromPath(m_TrainPath + "Model/");

	vector<Mat_uc>			ImageVec;
	vector<Mat_d>			TruthShapeVec;
	vector<cv::Rect>		BoxVec;
	LoadImageList(ImageListPath, ImageVec, TruthShapeVec, BoxVec);

	if (!ImageVec.empty())
	{
		double_t E = 0;
		int32_t WaitTime = -1;
		for (int32_t i = 0; i < ImageVec.size(); ++i)
		{
			Mat_d PredictShape = Predict(ImageVec[i], BoxVec[i]);
			double_t Err = CalculateError(TruthShapeVec[i], PredictShape);
			E += Err;
			if (WaitTime == -1)
			{
				Mat_d ImagePredictShape = Coordinate::Box2Image(PredictShape, BoxVec[i]);
				std::cout << Err << std::endl;
				for (int32_t Landmark = 0; Landmark < ImagePredictShape.rows; ++Landmark)
				{
					cv::circle(ImageVec[i], { static_cast<int32_t>(ImagePredictShape(Landmark,0)) ,static_cast<int32_t>(ImagePredictShape(Landmark,1)) }, 2, { 255 }, -1);
				}
				cv::imshow("TestImage", ImageVec[i]);
				if (cv::waitKey(WaitTime) == 'a')
					WaitTime = 0;
			}
		}
		std::cout << "Error :[" << E << "]\t Mean Error :[" << E / ImageVec.size() << "]" << std::endl;
	}
	cv::VideoCapture Cam("K:\\3.mp4");

	cv::Mat ImageColor;
	cv::CascadeClassifier Cs("../OpenCV/etc/haarcascades/haarcascade_frontalface_alt.xml");

	vector<cv::Rect> LastFaceBoxs;
	Mat_d PredictShape;
	while (true)
	{
		Mat_uc Image;
		Cam >> ImageColor;
		cvtColor(ImageColor, Image, CV_BGR2GRAY);

		vector<cv::Rect> FaceBoxs;
		if (LastFaceBoxs.empty())
			Cs.detectMultiScale(Image, FaceBoxs);

		if (FaceBoxs.empty())
			FaceBoxs = LastFaceBoxs;
		else
			LastFaceBoxs = FaceBoxs;
		for (auto& var : FaceBoxs)
		{
			PredictShape = Predict(Image, var, PredictShape);
			Mat_d ImagePredictShape = Coordinate::Box2Image(PredictShape, var);
			for (int32_t Landmark = 0; Landmark < g_TrainParam.LandmarkNumPerFace; ++Landmark)
			{
				cv::circle(ImageColor, { static_cast<int32_t>(ImagePredictShape(Landmark,0)) ,static_cast<int32_t>(ImagePredictShape(Landmark,1)) }, 2, { 255 }, -1);
			}
			cv::rectangle(ImageColor, var, { 0 });
		}
		cv::imshow("T", ImageColor);
		if (cv::waitKey(30) == 'r')
		{
			LastFaceBoxs.clear();
			PredictShape = Mat_d();
		}
	}

}

Mat_d FgLBFTrain::Predict(Mat_uc & Image, cv::Rect Box, Mat_d& LastFrame)
{
	Mat_d TransformedMeanShape;
	if (LastFrame.empty())
		TransformedMeanShape = g_TrainParam.MeanShape;
	else
	{
		vector<cv::Point2d> TempVec;
		Mat_d Transform = FgGetAffineTransform(g_TrainParam.MeanShape, LastFrame);
		cv::transform(ShapeToVecPoint(g_TrainParam.MeanShape), TempVec, Transform);
		TransformedMeanShape = VecPointToShape(TempVec);

		{
			Mat_uc I = Image.clone();
			Mat_d InitShape = TransformedMeanShape.clone();
			InitShape = Coordinate::Box2Image(InitShape, Box);
			for (int32_t Landmark = 0; Landmark < g_TrainParam.LandmarkNumPerFace; ++Landmark)
			{
				cv::circle(I, { static_cast<int32_t>(InitShape(Landmark,0)) ,static_cast<int32_t>(InitShape(Landmark,1)) }, 2, { 255 }, -1);
			}
			cv::imshow("Init", I);
		}
	}
	Mat_d CurrentShape = TransformedMeanShape.clone();

	for (int32_t Stage = 0; Stage < m_RegressorVec.size(); ++Stage)
	{
		Mat_d MeanShapeTo = FgGetAffineTransform(g_TrainParam.MeanShape, CurrentShape);
		MeanShapeTo(0, 2) = MeanShapeTo(1, 2) = 0;
		CurrentShape += m_RegressorVec[Stage].Predict(Image, CurrentShape, Box, MeanShapeTo);
	}
	return CurrentShape;
}

void FgLBFTrain::LoadFromPath(string Path)
{
	std::ifstream ifs(Path + "Model.bin");
	if (!ifs.is_open())
		ThrowFaile;

	ifs >> g_TrainParam.LocalFeaturesNum;
	ifs >> g_TrainParam.LandmarkNumPerFace;
	ifs >> g_TrainParam.RegressroStage;
	ifs >> g_TrainParam.TreeDepth;
	ifs >> g_TrainParam.TreeNumPerForest;
	ifs >> g_TrainParam.MeanShape;
	ifs >> g_TrainParam.DataAugmentOverLap;
	ifs >> g_TrainParam.DataAugmentScale;

	g_TrainParam.LocalRadiusPerStageVec.resize(g_TrainParam.RegressroStage);
	for (auto& var : g_TrainParam.LocalRadiusPerStageVec)
		ifs >> var;

	m_RegressorVec.resize(g_TrainParam.RegressroStage, FgLBFRegressor(vector<FgFaceData>()));
	for (int32_t i = 0; i < g_TrainParam.RegressroStage; ++i)
		m_RegressorVec[i].LoadFromPath(Path, i);

	ifs.close();
	return;
}

void FgLBFTrain::SaveToPath(string Path)
{
	_mkdir(Path.c_str());

	std::ofstream ofs(Path + "Model.bin");
	if (!ofs.is_open())
		ThrowFaile;

	ofs << g_TrainParam.LocalFeaturesNum << std::endl;
	ofs << g_TrainParam.LandmarkNumPerFace << std::endl;
	ofs << g_TrainParam.RegressroStage << std::endl;
	ofs << g_TrainParam.TreeDepth << std::endl;
	ofs << g_TrainParam.TreeNumPerForest << std::endl;
	ofs << g_TrainParam.MeanShape;
	ofs << g_TrainParam.DataAugmentOverLap << std::endl;
	ofs << g_TrainParam.DataAugmentScale << std::endl;

	for (auto& var : g_TrainParam.LocalRadiusPerStageVec)
		ofs << var << std::endl;

	ofs.flush();
	ofs.close();

	int32_t Idx = 0;
	for (auto& var : m_RegressorVec)
		var.SaveToPath(Path, Idx++);

	return;
}

void FgLBFTrain::LoadImageList(string FilePath, vector<Mat_uc>& ImageVec, vector<Mat_d>& TruthShape, vector<cv::Rect>& BoxVec)
{
	std::ifstream fin(FilePath);
	if (!fin.is_open())
		return;

	vector<string> ImagePathVec;
	vector<string> PtsPathVec;
	vector<string> FgrPathVec;

	string ImagePath;
	while (std::getline(fin, ImagePath))
	{
		ImagePathVec.push_back(ImagePath);

		string PtsPath = ImagePath;
		auto Pos = PtsPath.find_last_of('.');
		PtsPath.replace(Pos, PtsPath.length() - 1, ".pts");

		string FgrPath = ImagePath;
		FgrPath.replace(Pos, FgrPath.length() - 1, ".fgr");

		PtsPathVec.push_back(PtsPath);
		FgrPathVec.push_back(FgrPath);
	}
	fin.close();

	std::mutex mtx;
	concurrency::parallel_for(0, static_cast<int32_t>(ImagePathVec.size()), [&](int32_t i)
	{
		Mat_d Shape(g_TrainParam.LandmarkNumPerFace, 2, 0.0);

		std::ifstream fpts(PtsPathVec[i]);
		if (fpts.is_open())
		{
			int32_t landmark;
			string unuse;

			std::getline(fpts, unuse);
			fpts >> unuse >> landmark;
			if (landmark == g_TrainParam.LandmarkNumPerFace)
			{
				std::getline(fpts, unuse);
				std::getline(fpts, unuse);
				for (int32_t i = 0; i < landmark; ++i)
					fpts >> Shape(i, 0) >> Shape(i, 1);
			}
			fpts.close();
		}

		Mat_uc Image = cv::imread(ImagePathVec[i], CV_LOAD_IMAGE_GRAYSCALE);
		if (Image.empty())
			ThrowFaile;

		std::cout << ImagePathVec[i] << std::endl;

		double MinX, MinY, MaxX, MaxY;
		cv::minMaxIdx(Shape.col(0), &MinX, &MaxX);
		cv::minMaxIdx(Shape.col(1), &MinY, &MaxY);
		cv::Point2i MinPoint = cv::Point2i(static_cast<int32_t>(MinX), static_cast<int32_t>(MinY));
		cv::Point2i MaxPoint = cv::Point2i(cvCeil(MaxX), cvCeil(MaxY));

		vector<cv::Rect> FaceBoxs;
		std::ifstream fgrs(FgrPathVec[i]);
		if (fgrs.is_open())
		{
			int32_t RectNum = 0;
			fgrs >> RectNum;
			FaceBoxs.resize(RectNum);
			for (auto& var : FaceBoxs)
				fgrs >> var.x >> var.y >> var.width >> var.height;
			fgrs.close();
		}

		for (const auto& var : FaceBoxs)
		{
			//scale 1.5
			double_t Scale = 1.5;
			cv::Rect ScaleRect;

			double_t CenterX = var.x + 0.5 * var.width;
			double_t CenterY = var.y + 0.5 * var.height;

			ScaleRect.x = std::max(static_cast<int32_t>(CenterX - Scale / 2 * var.width), 0);
			ScaleRect.y = std::max(static_cast<int32_t>(CenterY - Scale / 2 * var.height), 0);
			ScaleRect.width = std::min(static_cast<int32_t>(Scale * var.width), Image.cols - var.x);
			ScaleRect.height = std::min(static_cast<int32_t>(Scale * var.height), Image.rows - var.y);

			if (ScaleRect.contains(MinPoint) && ScaleRect.contains(MaxPoint))
			{
				std::lock_guard<std::mutex> Lock(mtx);
				ImageVec.push_back(Image);
				BoxVec.push_back(var);
				TruthShape.push_back(Coordinate::Image2Box(Shape, var));
				break;
			}
		}
	}
	);
}

void FgLBFTrain::DataAugment()
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<int32_t> RandomGen(0, static_cast<int32_t>(g_ImageVec.size() - 1));
	std::uniform_int_distribution<int32_t> RotateAngleGen(0, 30);
	std::uniform_real_distribution<double_t> ScaleGen(0.7, 1.3);

	for (int32_t i = 0; i < g_ImageVec.size(); ++i)
	{
		for (int32_t j = 0; j < g_TrainParam.DataAugmentScale; ++j)
		{
			int32_t idx = 0;
			do
			{
				idx = RandomGen(rng);
			} while (idx == i);

			FgFaceData temp;
			temp.BoxIdx = i;
			temp.ImageIdx = i;
			temp.TruthShapeIdx = i;
			temp.CurrentShape = g_TruthShapeVec[idx].clone();

			m_FaceDataVec.push_back(temp);
		}

		for (int32_t j = 0; j < g_TrainParam.DataAugmentScale; ++j)
		{
			Mat_d RotationMat = cv::getRotationMatrix2D({ 0,0 }, RotateAngleGen(rng), ScaleGen(rng));
			vector<cv::Point2d> TempVec;
			cv::transform(ShapeToVecPoint(g_TrainParam.MeanShape.clone()), TempVec, RotationMat);

			FgFaceData temp;
			temp.BoxIdx = i;
			temp.ImageIdx = i;
			temp.TruthShapeIdx = i;
			temp.CurrentShape = VecPointToShape(TempVec);
		}

		FgFaceData temp;
		temp.BoxIdx = i;
		temp.ImageIdx = i;
		temp.TruthShapeIdx = i;
		temp.CurrentShape = g_TrainParam.MeanShape.clone();

		m_FaceDataVec.push_back(temp);
	}
}
