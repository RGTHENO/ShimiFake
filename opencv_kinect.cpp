/*****
 * sean: that is a driver bug
 * check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger
 * 
 * 
 * ***/


#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/imgproc/types_c.h>

#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h" 
 

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>


using namespace std;
using namespace cv;

using namespace dlib;


enum
{
	Processor_cl,
	Processor_gl,
	Processor_cpu
};

bool protonect_shutdown = false; // Whether the running application should shut down.

void sigint_handler(int s)
{
	protonect_shutdown = true;
}

void canyEdgeDetector( Mat& rgb_image ){
	
	 Mat gray, edge, draw;
    cvtColor(rgb_image, gray, CV_BGR2GRAY);
    Canny( gray, edge, 50, 150, 3);
 
    edge.convertTo(draw, CV_8U); 
    imshow("image", draw);


}


bool is_closedEyes(std::vector<cv::Point>  vec ){ ///
 
	cv::Point middle = vec[vec.size()/2];
	
	for(auto p_i :vec ){
		
		if(abs (middle.y- p_i.y ) > 5 ) ///Dsitancia de MANHATAN
			 return false;
	}	
	return true;
}

void segmentarRostros( Mat& img, std::vector<cv::Rect> &facesBoxes){
		
		cv::Mat frame_gray;
		cv::cvtColor(img, frame_gray, CV_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);
        const rgb_pixel color(255, 255, 0);
        const cv::Scalar scalar(255, 255, 0);
        std::vector<dlib::rectangle> facesRect_DLIB; ///Es importante crear otro vector de rectangles (en DLIB a diferencia de OpenCV) porque los algoritmos de landmarks están en DLIB
        /// guardar la region en donde se encuentra la cara
        
        
        image_window win;
        win.set_size(420, 380);
        win.set_title("Face Landmark Detector :: Tutor de Programacion");

        
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
        
        
        for (cv::Rect& rc : facesBoxes) { /// facesBoxes cv:Rect
                ///cv::rectangle(temp, rc, scalar, 1, cv::LINE_AA);
                facesRect_DLIB.push_back(dlib::rectangle(rc.x, rc.y, rc.x + rc.width, rc.y + rc.height)); ///facesRect desde DLIB
         }
         
         cv_image<bgr_pixel> cimg(img); 		
		 ///string ty =  type2str( temp.type() );
		 ///printf("Matrix: %s %dx%d \n", ty.c_str(), temp.cols,temp.rows );
		 cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC3 );
            
         cv_image<bgr_pixel> cimg_mask(mask);
         
 		 /// guarda los puntos obtenidos
		 std::vector<cv::Point> points_Landmarks;
		 std::vector<image_window::overlay_circle> points_Circles;
		 std::vector<full_object_detection> detects;
		 std::vector<cv::Point> nariz;
		 std::vector<cv::Point> eyeRight; /// 37 -- 42
		 std::vector<cv::Point> eyeLeft;     /// 43 -- 48
 
         /// detectaremos los landmarks para cada rostro encontrado
         for (unsigned long i = 0; i < facesRect_DLIB.size(); ++i) {
            
                full_object_detection shape = pose_model(cimg, facesRect_DLIB[i]);
                detects.push_back(shape);
                
				///cout<<"pixel 36 part en"<<i<<" es :"<<shape.part(36)<<endl;
				///cout<<"pixel 36 part en"<<i<<" es :"<<shape.part(37)<<endl;
				///cout<<"pixel 36 part en"<<i<<" es :"<<shape.part(38)<<endl;
                
                /// guardaremos las coordenada de cada landmark
                for (unsigned int n = 0; n < shape.num_parts(); n++) {
                    
                    point pt = shape.part(n); ///Se recupera el landmark n-esimo [posicion]
                    
                    points_Circles.push_back(image_window::overlay_circle(pt, 2, color)); ///el vector "points" contiene todos los circles de los ptos de los landmarks
					points_Landmarks.push_back(cv::Point(pt.x(),pt.y() ));	
					
					/**** Vamos procesar los landmarks de la nariz y de los ojos ****/	
		
					if(n>=38 && n<=42){
						
					    eyeRight.push_back(cv::Point(pt.x(), pt.y()));
                        points_Circles.push_back(image_window::overlay_circle(pt, 2, rgb_pixel(128+n,n,200+n) ));

					}
					else if( n>=44 && n<48){
						eyeLeft.push_back(cv::Point(pt.x(), pt.y()));
                        points_Circles.push_back(image_window::overlay_circle(pt, 2, rgb_pixel(128+n,n,200+n) ));
					}
					  
                    else if (n >= 28 && n < 36) {
                        nariz.push_back(cv::Point(pt.x(), pt.y()));
                        ///cout<<"Puntos de la nariz: "<<pt.x()<<", "<<pt.y()<<endl;
                    }
                    else{
						points_Circles.push_back(image_window::overlay_circle(pt, 2, color));
					} 
					/**** Fin de de procesar los landmarks de nariz y de los ojos ****/
                }
                
                std::vector< cv::Point > hull(points_Landmarks.size());    /// Hull: contendrá los puntos landmarks que contornean el rostro
				
				convexHull(cv::Mat(points_Landmarks), hull, false);        /// Calcularemos esos puntos landmarks que contornean el rostro   
                polylines(img,hull,true,cv::Scalar(255,255,255),2,150,0); /// Dibujo sobre temp, el poligono formado por todos
																		   /// los puntos que contornean el rostro.
                
                cv::fillConvexPoly(mask,   ///Imagen sobre el cual se va a dibujar la mascara
                   hull, cv::Scalar(255,255,255), 16,0);
                
                cv::bitwise_and(img, mask, img);   /// Segmentamos el rostro aplicando la mascara "mask"
													 /// y lo almacenamos en "mask" que tiene las mismas
													 /// dimensiones que tiene temp(source image)
                
                cv_image<bgr_pixel> cimg_mask(mask); ///Obtenemos una referencia a nuestra mascara "mask" 
       
                /**  
				cv::Point middle = eyeLeft[eyeLeft.size()/2];
	
				for(auto p_i :eyeLeft ){		
					cout<<"DISTANCIA ojo izq:"<<middle.y- p_i.y<<endl;			
				}
	            **/
	
	
                /// /**
				///printVectorPoint(eyeRight, "ojoDerecho");
				///printVectorPoint(eyeLeft, "ojoIzquierdo");
					
				string rpta ="";
                if( is_closedEyes(eyeRight) )
					rpta = "D";
                else{ 
					///cout<<"CASO OJO IZQ\n";
					if(is_closedEyes(eyeLeft) ){
						///rpta = "izq cerrado";
						rpta = "I";
					}
					else{
						///cout<<"OJO IZQ ABIERTO y derecho abierto\n";	
						rpta = "N";
					}
				} 
					
                ///  **/         
      	
				/*** INICIO : Conexion usando sockets **/
		        ///rpta = "Derecha";
		        /***sockets
				n = write(SocketFD, rpta.c_str(), rpta.length());
				memset(&rpta, 0, sizeof(rpta));
				
				bzero(buffer,256);
				n = read(SocketFD,buffer, 255);
				printf("Server:%s \n",buffer); 
				
				
				/************ FIN: conexion usando sockets***************/
				
                
            }
            
            
            

            /// dibujar la region de la nariz
            ///cv::Rect rectNariz = cv::boundingRect(nariz);
            ///cv::rectangle(temp, rectNariz, cv::Scalar(0, 0, 255));

            // Display it all on the screen
             win.clear_overlay();            
             win.set_image(cimg_mask);                              // establecer la imagen en la ventana
             ///cv::imshow("la mascara", mask);
             ///win.add_overlay(points);                          // mostrar los puntos de los landmarks
             ///win.add_overlay(render_face_detections(detects)); // dibujar los landmarks como lineas 
           
} ///Fin while


 
void detectorRostros(Mat& rgb_image, CascadeClassifier& detector, Mat& aux, std::vector<cv::Rect> &rect){  ///"rect" contiene todos los bounding boxes con los rostros detectados
	
	Mat gray, dest;
    ///cvtColor(rgb_image, gray, CV_BGR2GRAY);
	cvtColor(aux, gray, CV_BGR2GRAY);
	
	equalizeHist(gray, dest);

	///vector<Rect> rect;
		
	detector.detectMultiScale(dest, rect, 1.2, 3, 0, Size(60,60));
	///cv::imshow("depth2RGB deteccion de rostros", aux/4500.0f);
	
	for(Rect rc : rect)
	{
		cout<<"rc.x :"<<rc.x<<" - "<<"rc.y :"<<rc.y<<endl;
		cv::rectangle(aux, 
			Point(rc.x, rc.y), 
			Point(rc.x + rc.width, rc.y + rc.height), 
			CV_RGB(0,255,0), 2);
			
		/**rectangle(aux, 
			Point(rc.x, rc.y), 
			Point(rc.x + rc.width, rc.y + rc.height), 
			CV_RGB(0,255,0), 2);
		***/
	}
	
	/***Vamos a crear una imagen con datos de profunidad que solo muestre el rostro detectado*****/
	
	///Mat depthMap_face;
	
	if( aux.rows == rgb_image.rows && aux.cols == rgb_image.cols){
		cout<<"Son iguales csm!\n";
	} else{
		cout<<"NO son iguaels\n";
		cout<<"depthmap.rows: "<<aux.rows<<" y "<<"depthmap.cols : "<<aux.cols<<endl;
		cout<<"rgb_image.rows: "<<rgb_image.rows<<" y "<<"rgb_image.cols : "<<rgb_image.cols<<endl;
	}

    /***
	imshow("Deteccion de rostros", rgb_image);
	imshow("Deteccion de rostros sobre la imagen de profundidad", aux);
	**/
	imshow("Deteccion de rostros sobre la imagen de profundidad", aux);
	

}


int main()
{
	std::cout << "Hello World!" << std::endl;

	libfreenect2::Freenect2 freenect2;
	libfreenect2::Freenect2Device *dev = NULL;
	libfreenect2::PacketPipeline  *pipeline = NULL;

	if(freenect2.enumerateDevices() == 0)
	{
		std::cout << "no device connected!" << std::endl;
		return -1;
	}

	string serial = freenect2.getDefaultDeviceSerialNumber();

	std::cout << "SERIAL: " << serial << std::endl;

#if 1 // sean
	int depthProcessor = Processor_cl;

	if(depthProcessor == Processor_cpu)
	{
		if(!pipeline)
			//! [pipeline]
			pipeline = new libfreenect2::CpuPacketPipeline();
		//! [pipeline]
	}
	else if (depthProcessor == Processor_gl) // if support gl
	{
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
		if(!pipeline)
		{
			pipeline = new libfreenect2::OpenGLPacketPipeline();
		}
#else
		std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
	}
	else if (depthProcessor == Processor_cl) // if support cl
	{
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
		if(!pipeline)
			pipeline = new libfreenect2::OpenCLPacketPipeline();
#else
		std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
	}

	if(pipeline)
	{
		dev = freenect2.openDevice(serial, pipeline);
	}
	else
	{
		dev = freenect2.openDevice(serial);
	}

	if(dev == 0)
	{
		std::cout << "failure opening device!" << std::endl;
		return -1;
	}

	signal(SIGINT, sigint_handler);
	protonect_shutdown = false;

	libfreenect2::SyncMultiFrameListener listener(
			libfreenect2::Frame::Color |
			libfreenect2::Frame::Depth |
			libfreenect2::Frame::Ir);
	libfreenect2::FrameMap frames;

	dev->setColorFrameListener(&listener);
	dev->setIrAndDepthFrameListener(&listener);

	dev->start();

	std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
	std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

	libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
	libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); ///undistorted is just the rectified depth image. registered is the color image mapped onto depth.
	///libfreenect2::Frame undistorted(1920, 1080, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4);  ////Por que +2 en el valor del numero de filas ???????
	

	Mat rgbmat, depthmat, depthmatUndistorted, irmat, rgbd, rgbd2;

	cv::namedWindow("rgb", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("ir", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("depth", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("undistorted", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("registered", WND_PROP_ASPECT_RATIO);
	cv::namedWindow("depth2RGB", WND_PROP_ASPECT_RATIO);
	
	CascadeClassifier detector;
	if(!detector.load("haarcascade_frontalface_alt.xml")) 
		cout << "No se puede abrir clasificador." << endl;

	while(!protonect_shutdown)
	{
		listener.waitForNewFrame(frames);
		libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
		libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
		libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

		cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbmat);
		cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irmat);
		cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);

		cv::imshow("rgb", rgbmat);
		cv::imshow("ir", irmat / 4500.0f);
		cv::imshow("depth", depthmat / 4500.0f);
		
		
		/*******  Procesamiento RGB image ***/
   
		canyEdgeDetector(rgbmat);
		///detectorRostros(rgbmat, detector, depthmat);
  
		/**************************************/

		registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);

		cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(depthmatUndistorted);
		cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd);
		cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(rgbd2);

		cv::imshow("undistorted", depthmatUndistorted / 4500.0f);
		cv::imshow("registered", rgbd);
		cv::imshow("depth2RGB", rgbd2 / 4500.0f);
		
		std::vector<cv::Rect> facesBoxes;
		detectorRostros(rgbmat, detector, rgbd, facesBoxes);	
		
		///segmentRostros(    );
	
		

		int key = cv::waitKey(1);
		protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

		listener.release(frames);
	}

	dev->stop();
	dev->close();

	delete registration;

#endif

	std::cout << "Goodbye World!" << std::endl;
	return 0;
}
