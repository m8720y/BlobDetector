BlobDetector bd;

void setup()
{
  bd = new BlobDetector(this);
  bd.init("Webcam");
  //bd.init("Kinect");
}

void draw()
{
  bd.detect();
  bd.drawBlob();
  //bd.drawContour();
  
//  for(int i = 0; i < bd.getBlobNum(); i++)
//  {
//    Blob b = bd.getBlob(i);
//    float xMid = (b.xMin + b.xMax)/2;
//    float yMid = (b.yMin + b.yMax)/2;
//    int areaIndex = bd.getAreaIndex(xMid, yMid);
//    //println(xMid, yMid, areaIndex);
//    
//  }
}


void keyPressed() {
  bd.setInitImage();
}

