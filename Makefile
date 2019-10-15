all:opencv_kinect
CFLAGS=-fPIC -g -Wall `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
INCLUDE = -I/usr/local/include/libfreenect2 -I/usr/local/include/opencv4/
FREE_LIBS = -L/usr/local/lib -lfreenect2 -L /usr/local/lib -lopencv_core -lopencv_highgui
opencv_kinect:opencv_kinect.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $? -o $@  $(LIBS) $(FREE_LIBS) -ldlib

%.o:%.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean:
	rm -rf *.o opencv_kinect
