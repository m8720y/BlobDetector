import blobDetection.*;

import org.opencv.core.*;
import gab.opencv.*;
//import java.nio.*;

import KinectPV2.*;
import processing.video.*;
import org.opencv.highgui.VideoCapture;

class BlobDetector {
  sketch_bd m_parent;
  OpenCV m_opencv;
  KinectPV2 m_kinect;
  Capture m_webcam;
  BlobDetection m_blobDetection;
  boolean m_isInitImage = false;
  boolean m_isKinect = false;

  PImage m_currentImage;
  PImage m_initImage;
  PImage m_grayDiffImage;
  PImage m_areaImage;
  PImage m_contourImage;

  int m_depthImageWidth = 512;
  int m_depthImageHeight = 424;

  public BlobDetector(sketch_bd bd) {
    this.m_parent = bd;
  }

  void init(String cam)
  {
    if (cam == "Kinect") {
      m_isKinect = true;
    } else {
      m_isKinect = false;
    }
    init();
  }

  void init()
  {
    size(m_depthImageWidth * 2, m_depthImageHeight);

    if (m_isKinect) {
      setupKinect();
    } else {
      setupWebcam();
    }

    // BlobDetection
    m_isInitImage = true;
    m_currentImage  = createImage(m_depthImageWidth, m_depthImageHeight, RGB);
    m_initImage     = createImage(m_depthImageWidth, m_depthImageHeight, RGB);
    m_grayDiffImage = createImage(m_depthImageWidth, m_depthImageHeight, RGB);
    m_contourImage = createImage(m_depthImageWidth, m_depthImageHeight, RGB);
    m_areaImage = loadImage("./image.bmp");

    m_blobDetection = new BlobDetection(m_depthImageWidth, m_depthImageHeight);
    m_blobDetection.setPosDiscrimination(true);
    m_blobDetection.setThreshold(0.2f); // will detect bright areas whose luminosity > 0.2f;
  }

  void setInitImage()
  {
    if (m_isInitImage == false) {
      m_isInitImage = true;
      println("press key");
    }
  }

  void setupKinect()
  {
    m_kinect = new KinectPV2(m_parent);
    m_kinect.enableDepthImg(true);
    m_kinect.enablePointCloud(true);
    m_kinect.enableColorImg(true);
    m_kinect.init();
  }

  void setupWebcam()
  {
    String[] cameras = Capture.list();
    m_webcam = new Capture(m_parent, cameras[0]);
    m_webcam.start();
  }

  void getKinectImage()
  {
    m_opencv = new OpenCV(m_parent, m_depthImageWidth, m_depthImageHeight);
    m_opencv.loadImage(m_kinect.getPointCloudDepthImage());
    m_kinect.setLowThresholdPC (0.5f);
    m_kinect.setHighThresholdPC(5.0f);

    if (m_isInitImage) {
      m_isInitImage = false;
      image(m_kinect.getPointCloudDepthImage(), 0, 0, m_kinect.getDepthImage().width, m_kinect.getDepthImage().height);
      m_initImage.copy(m_opencv.getOutput(), 0, 0, m_kinect.getDepthImage().width, m_kinect.getDepthImage().height, 
      0, 0, m_depthImageWidth, m_depthImageHeight);
    }

    m_currentImage.copy(m_opencv.getOutput(), 0, 0, m_kinect.getDepthImage().width, m_kinect.getDepthImage().height, 
    0, 0, m_depthImageWidth, m_depthImageHeight);

    image(m_kinect.getColorImage(), m_depthImageWidth, 0, m_depthImageWidth, m_depthImageWidth * 360 / 640);
  }

  void getWebcamImage()
  {
    m_opencv = new OpenCV(m_parent, m_depthImageWidth, m_depthImageHeight);
    m_webcam.read();
    if (m_isInitImage)
    {
      m_isInitImage = false;
      image(m_webcam, 0, 0, m_webcam.width, m_webcam.width);
      m_initImage.copy(m_webcam, 0, 0, m_webcam.width, m_webcam.height, 
      0, 0, m_currentImage.width, m_currentImage.height);
    }

    image(m_webcam, 0, 0, m_webcam.width, m_webcam.width);
    m_currentImage.copy(m_webcam, 0, 0, m_webcam.width, m_webcam.width, 
    0, 0, m_currentImage.width, m_currentImage.height);

    image(m_currentImage, m_depthImageWidth, 0, m_depthImageWidth, m_depthImageHeight);
  }  

  void detect()
  {
    if (m_isKinect) {
      getKinectImage();
    } else {
      getWebcamImage();
    }
    imageSubtraction(m_grayDiffImage, m_initImage, m_currentImage);
    fastblur(m_grayDiffImage, 5);
    image(m_grayDiffImage, 0, 0);
    m_blobDetection.computeBlobs(m_grayDiffImage.pixels);
  }
  
  void drawContour()
  {
    m_opencv.loadImage(m_grayDiffImage);
    m_opencv.dilate();
    m_opencv.erode();
    m_opencv.contrast(100);
    m_opencv.brightness(-10);
  
    fill(255);
    noStroke();
    for (Contour contour : m_opencv.findContours()) {
      contour = contour.getConvexHull();
      if(contour.area() > 1000){
        println(contour.area());
        contour.draw();
      }
    }

    noFill();
    stroke(255, 0, 0);
    strokeWeight(3);
    m_opencv.loadImage(m_opencv.getSnapshot());
    for (Contour contour : m_opencv.findContours()) {
      if(contour.area() > 1000){//getBoundingBox
        line(contour.getBoundingBox().x,
        contour.getBoundingBox().y, 
        contour.getBoundingBox().x + contour.getBoundingBox().width, 
        contour.getBoundingBox().y);
        contour.draw();
      }
    }
  }

  void drawBlob()
  {
    noFill();
    Blob b;
    EdgeVertex eA, eB;
    for (int i = 0; i < m_blobDetection.getBlobNb (); i++)
    {
      b = m_blobDetection.getBlob(i);
      if (b != null)
      {
        // Edges
        strokeWeight(1);
        stroke(0, 255, 0);
        for (int j = 0; j < b.getEdgeNb (); j++)
        {
          eA = b.getEdgeVertexA(j);
          eB = b.getEdgeVertexB(j);
          if (eA !=null && eB !=null )
            line(
            eA.x * m_depthImageWidth, eA.y * m_depthImageHeight, 
            eB.x * m_depthImageWidth, eB.y * m_depthImageHeight
           );
        }
        

        // Draw Blobs
        if (b.w > 0.1 && b.h > 0.1)
        {
          // area Index
          int areaIndex = getAreaIndex((b.xMin + b.xMax)/2, (b.yMin + b.yMax)/2);
          textSize(32);
          text(areaIndex, b.xMin * m_depthImageWidth, b.yMin * m_depthImageHeight + 32);
          
          strokeWeight(10);
          stroke(255, 255, 0);
          point((b.xMin + b.xMax)/2 * m_depthImageWidth, (b.yMin + b.yMax)/2 * m_depthImageHeight);
        
          // stroke
          strokeWeight(5);
          stroke(255, 0, 0);
          rect(b.xMin * m_depthImageWidth, b.yMin * m_depthImageHeight, 
          b.w * m_depthImageWidth, b.h * m_depthImageHeight
            );
        }
      }
    }
  }

  Blob getBlob(int i)
  {
    return m_blobDetection.getBlob(i);
  }

  int getBlobNum()
  {
    return m_blobDetection.getBlobNb ();
  }
  
  int getAreaIndex(float x, float y)
  {
    int intX = int(x * m_depthImageWidth);
    int intY = int(y * m_depthImageHeight);
    int index = intY * m_depthImageWidth + intX;
    float r = red(m_areaImage.pixels[index]);
    
    
    int areaIndex = 0;
    if(r > 220) areaIndex = 1;
    else if(r > 190) areaIndex = 2;
    else if(r > 170) areaIndex = 3;
    else if(r > 150) areaIndex = 4;
    else if(r > 130) areaIndex = 5;
    else if(r > 110) areaIndex = 6;
    
    return areaIndex;
  }

  void fastblur(PImage img, int radius)
  {
    if (radius<1) {
      return;
    }
    int w=img.width;
    int h=img.height;
    int wm=w-1;
    int hm=h-1;
    int wh=w*h;
    int div=radius+radius+1;
    int r[]=new int[wh];
    int g[]=new int[wh];
    int b[]=new int[wh];
    int rsum, gsum, bsum, x, y, i, p, p1, p2, yp, yi, yw;
    int vmin[] = new int[max(w, h)];
    int vmax[] = new int[max(w, h)];
    int[] pix=img.pixels;
    int dv[]=new int[256*div];
    for (i=0; i<256*div; i++) {
      dv[i]=(i/div);
    }

    yw=yi=0;

    for (y=0; y<h; y++) {
      rsum=gsum=bsum=0;
      for (i=-radius; i<=radius; i++) {
        p=pix[yi+min(wm, max(i, 0))];
        rsum+=(p & 0xff0000)>>16;
        gsum+=(p & 0x00ff00)>>8;
        bsum+= p & 0x0000ff;
      }
      for (x=0; x<w; x++) {

        r[yi]=dv[rsum];
        g[yi]=dv[gsum];
        b[yi]=dv[bsum];

        if (y==0) {
          vmin[x]=min(x+radius+1, wm);
          vmax[x]=max(x-radius, 0);
        }
        p1=pix[yw+vmin[x]];
        p2=pix[yw+vmax[x]];

        rsum+=((p1 & 0xff0000)-(p2 & 0xff0000))>>16;
        gsum+=((p1 & 0x00ff00)-(p2 & 0x00ff00))>>8;
        bsum+= (p1 & 0x0000ff)-(p2 & 0x0000ff);
        yi++;
      }
      yw+=w;
    }

    for (x=0; x<w; x++) {
      rsum=gsum=bsum=0;
      yp=-radius*w;
      for (i=-radius; i<=radius; i++) {
        yi=max(0, yp)+x;
        rsum+=r[yi];
        gsum+=g[yi];
        bsum+=b[yi];
        yp+=w;
      }
      yi=x;
      for (y=0; y<h; y++) {
        pix[yi]=0xff000000 | (dv[rsum]<<16) | (dv[gsum]<<8) | dv[bsum];
        if (x==0) {
          vmin[y]=min(y+radius+1, hm)*w;
          vmax[y]=max(y-radius, 0)*w;
        }
        p1=x+vmin[y];
        p2=x+vmax[y];

        rsum+=r[p1]-r[p2];
        gsum+=g[p1]-g[p2];
        bsum+=b[p1]-b[p2];

        yi+=w;
      }
    }
  }


  void imageSubtraction(PImage imgOut, PImage img1, PImage img2) {
    for (int i=0; i < img1.pixels.length; i++) {


      float gray1 = (red(img1.pixels[i]) + green(img1.pixels[i]) + blue(img1.pixels[i])) / 3.0;
      float gray2 = (red(img2.pixels[i]) + green(img2.pixels[i]) + blue(img2.pixels[i])) / 3.0;

      float diff = abs(gray1 - gray2);
      //if (i % m_depthImageWidth < 30) diff = 0;
      if (diff < 30) diff = 0; 

      imgOut.pixels[i] = color(diff, diff, diff);
    }

    imgOut.updatePixels();
  }
}
