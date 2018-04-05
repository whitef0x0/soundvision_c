/* C program for soundscape generation. (C) P.B.L. Meijer 1996 */
/* hificode.c modified for camera input using OpenCV. (C) 2013 */
/* Last update: March 19, 2016; released under the Creative    */
/* Commons Attribution 4.0 International License (CC BY 4.0),  */
/* see http://www.seeingwithsound.com/im2sound.htm for details */ 

#define PY_ARRAY_UNIQUE_SYMBOL soundvision_c_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#define FNAME "/Users/admin/UBC/CPEN391/project2/tracking/hificode.wav"    /* User-defined parameters   */
#define FL     500   /* Lowest  frequency (Hz) in soundscape   */
#define FH    5000   /* Highest frequency (Hz)                 */
#define FS   44100   /* Sample  frequency (Hz)                 */
#define T     1.05   /* Image to sound conversion time (s)     */
#define D        1   /* Linear|Exponential=0|1 distribution    */
#define HIFI     1   /* 8-bit|16-bit=0|1 sound quality         */
#define STEREO   1   /* Mono|Stereo=0|1 sound selection        */
#define DELAY    1   /* Nodelay|Delay=0|1 model   (STEREO=1)   */
#define FADE     1   /* Relative fade No|Yes=0|1  (STEREO=1)   */
#define DIFFR    1   /* Diffraction No|Yes=0|1    (STEREO=1)   */
#define BSPL     1   /* Rectangular|B-spline=0|1 time window   */
#define BW       0   /* 16|2-level=0|1 gray format in *P[]     */
#define CAM      1   /* Use OpenCV camera input No|Yes=0|1     */
#define VIEW     1   /* Screen view for debugging No|Yes=0|1   */
#define CONTRAST 2   /* Contrast enhancement, 0=none           */

#define C_ALLOC(number, type) ((type *) calloc((number),sizeof(type)) )
#define TwoPi 6.283185307179586476925287
#define HIST  (1+HIFI)*(1+STEREO)
#define WHITE 1.00
#define BLACK 0.00

/* Soundscape resolution M rows x N columns */
#if CAM  
/* 176 x 64 for live camera view */
#define M     64
#define N    176
#else  
/* 64 x 64 for hard-coded image */
#define M     64
#define N     64
#endif

FILE *fp; 
unsigned long ir=0L, ia=9301L, ic=49297L, im=233280L;
void wi(unsigned int i) { int b1,b0; b0=i%256; b1=(i-b0)/256; putc(b0,fp); putc(b1,fp); }
void wl(long l) { unsigned int i1,i0; i0=l%65536L; i1=(l-i0)/65536L; wi(i0); wi(i1); }
double rnd(void){ ir = (ir*ia+ic) % im; return ir / (1.0*im); }

namespace soundvision_c {

   using namespace boost::python;

   int playSound(char *filename) {  /* Play soundscape */
      char command[256] = "";
      sprintf(command, "afplay %s", filename);
      int status = system(command);
      return status; 
   }

   // PyObject *readCamera(){
   //    cap.open(0);
   //    if (!cap.isOpened()) {
   //       fprintf(stderr,"Could not open camera %d\n", cam_id);
   //       return 1;
   //    }
   //    /* Setting standard capture size, may fail; resize later */
   //    cap.read(frame);   Dummy read needed with some devices 
   //    cap.set(CV_CAP_PROP_FRAME_WIDTH , 176);
   //    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 144);

   //    if (VIEW) {  /* Screen views only for debugging */
   //       namedWindow("Large", CV_WINDOW_AUTOSIZE);
   //       namedWindow("Small", CV_WINDOW_AUTOSIZE);
   //    }
   // }

   int i, j, d=D, ss, key=0;
   long k=0L, l, ns=2L*(long)(0.5*FS*T), m=ns/N,
      sso=HIFI?0L:128L, ssm=HIFI?32768L:128L;
   double a, t, dt=1.0/FS, s, y, yp, z, tau1, tau2, x, theta,
      scale=0.5/sqrt((double)M), q, q2, r, sl, sr, tl, tr, yl, ypl, yr, ypr,
      zl, zr, hrtf, hrtfl, hrtfr, v=340.0,  /* v = speed of sound (m/s) */
      hs=0.20;  /* hs = characteristic acoustical size of head (m) */
   double *w    = C_ALLOC(M, double);
   double *phi0 = C_ALLOC(M, double);
   double **A = C_ALLOC(M, double *);

   void setup(){
      for (i=0; i<M; i++) {
         A[i] = C_ALLOC(N, double);  /* M x N pixel matrix */
      }

      /* Set lin|exp (0|1) frequency distribution and random initial phase */
      if (d) {
         for (i=0; i<M; i++) {
            w[i] = TwoPi * FL * pow(1.0* FH/FL,1.0*i/(M-1));
         }
      } else {
         for (i=0; i<M; i++) {
            w[i] = TwoPi * FL + TwoPi * (FH-FL) * i/(M-1);
         }
      }

      for (i=0; i<M; i++){
         phi0[i] = TwoPi * rnd();
      }
   }

   void process_frame(PyObject *frameNDArray) {
      cv::Mat frame, gray;
      frame = soundvision_c::fromNDArrayToMat(frameNDArray);
      cv::Mat tmp;
      cv::cvtColor(frame,tmp,CV_BGR2GRAY);
      if (frame.rows != M || frame.cols != N) cv::resize(tmp, gray, cv::Size(N,M)); else gray=tmp;

      if (CAM) {  /* Set live camera image */
         int px;
         double avg;
         avg = 0.0;
         for (i=0; i<M; i++) {
            for (j=0; j<N; j++) {
               avg += gray.at<uchar>(M-1-i,j);
            }
         }
         avg = avg / (N * M);
         for (i=0; i<M; i++) {
            for (j=0; j<N; j++) {
               px = gray.at<uchar>(M-1-i,j);
               px += CONTRAST*(px - avg);
               if (px > 255) px = 255;
               if (px <   0) px =   0;
               gray.at<uchar>(M-1-i,j) = px;
               if (px == 0) A[i][j] = 0; 
               else A[i][j] = pow(10.0,(px/16-15)/10.0);  /* 2dB steps */
            }
         }
      }

      if (VIEW) {  /* Screen views only for debugging */
      /* imwrite("hificodeLarge.jpg", frame); */
         cv::imshow("Large", frame);
         cv::moveWindow("Large", 20, 20);
      /* imwrite("hificodeSmall.jpg", gray); */
         cv::imshow("Small", gray);
         cv::moveWindow("Small", 220, 20);
      }

      key = cv::waitKey((int)(10));

      /* Write 8/16-bit mono/stereo .wav file */
      fp = fopen(FNAME,"wb"); fprintf(fp,"RIFF"); wl(ns*HIST+36L);
      fprintf(fp,"WAVEfmt "); wl(16L); wi(1); wi(STEREO?2:1); wl(0L+FS);
      wl(0L+FS*HIST); wi(HIST); wi(HIFI?16:8); fprintf(fp,"data"); wl(ns*HIST);
      tau1 = 0.5 / w[M-1]; tau2 = 0.25 * tau1*tau1;
      y = yl = yr = z = zl = zr = 0.0;
      /* Not optimized for speed */
      while (k < ns && !STEREO) {
         if (BSPL) { q = 1.0 * (k%m)/(m-1); q2 = 0.5*q*q; }
         j = k / m; if (j>N-1) j=N-1; s = 0.0; t = k * dt;
         if (k < ns/(5*N)) s = (2.0*rnd()-1.0) / scale;  /* "click" */
         else for (i=0; i<M; i++) {
            if (BSPL) {  /* Quadratic B-spline for smooth C1 time window */
               if (j==0) a = (1.0-q2)*A[i][j]+q2*A[i][j+1];
               else if (j==N-1) a = (q2-q+0.5)*A[i][j-1]+(0.5+q-q2)*A[i][j];
               else a = (q2-q+0.5)*A[i][j-1]+(0.5+q-q*q)*A[i][j]+q2*A[i][j+1];
            }
            else a = A[i][j];  /* Rectangular time window */
            s += a * sin(w[i] * t + phi0[i]);
         }
         yp = y; y = tau1/dt + tau2/(dt*dt);
         y  = (s + y * yp + tau2/dt * z) / (1.0 + y); z = (y - yp) / dt;
         l  = sso + 0.5 + scale * ssm * y; /* y = 2nd order filtered s */
         if (l >= sso-1+ssm) l = sso-1+ssm; if (l < sso-ssm) l = sso-ssm;
         ss = (unsigned int) l;
         if (HIFI) wi(ss); else putc(ss,fp);
         k++;
      }
      while (k < ns && STEREO) {
         if (BSPL) { q = 1.0 * (k%m)/(m-1); q2 = 0.5*q*q; }
         j = k / m; if (j>N-1) j=N-1;
         r = 1.0 * k/(ns-1);  /* Binaural attenuation/delay parameter */
         theta = (r-0.5) * TwoPi/3; x = 0.5 * hs * (theta + sin(theta));
         tl = tr = k * dt; if (DELAY) tr += x / v; /* Time delay model */
         x  = fabs(x); sl = sr = 0.0; hrtfl = hrtfr = 1.0;
         for (i=0; i<M; i++) {
            if (DIFFR) {
               /* First order frequency-dependent azimuth diffraction model */
               if (TwoPi*v/w[i] > x) hrtf = 1.0; else hrtf = TwoPi*v/(x*w[i]);
               if (theta < 0.0) { hrtfl =  1.0; hrtfr = hrtf; }
               else             { hrtfl = hrtf; hrtfr =  1.0; }
            }
            if (FADE) {
               /* Simple frequency-independent relative fade model */
               hrtfl *= (1.0-0.7*r);
               hrtfr *= (0.3+0.7*r);
            }
            if (BSPL) {
               if (j==0) a = (1.0-q2)*A[i][j]+q2*A[i][j+1];
               else if (j==N-1) a = (q2-q+0.5)*A[i][j-1]+(0.5+q-q2)*A[i][j];
               else a = (q2-q+0.5)*A[i][j-1]+(0.5+q-q*q)*A[i][j]+q2*A[i][j+1];
            }
            else a = A[i][j];
            sl += hrtfl * a * sin(w[i] * tl + phi0[i]);
            sr += hrtfr * a * sin(w[i] * tr + phi0[i]);
         }
         if (k < ns/(5*N)) sl = (2.0*rnd()-1.0) / scale;  /* Left "click" */
         if (tl < 0.0) sl = 0.0;
         if (tr < 0.0) sr = 0.0;
         ypl = yl; yl = tau1/dt + tau2/(dt*dt);
         yl  = (sl + yl * ypl + tau2/dt * zl) / (1.0 + yl); zl = (yl - ypl) / dt;
         ypr = yr; yr = tau1/dt + tau2/(dt*dt);
         yr  = (sr + yr * ypr + tau2/dt * zr) / (1.0 + yr); zr = (yr - ypr) / dt;
         l   = sso + 0.5 + scale * ssm * yl;
         if (l >= sso-1+ssm) l = sso-1+ssm; if (l < sso-ssm) l = sso-ssm;
         ss  = (unsigned int) l;
         if (HIFI) wi(ss); else putc(ss,fp);  /* Left channel */
         l   = sso + 0.5 + scale * ssm * yr;
         if (l >= sso-1+ssm) l = sso-1+ssm; if (l < sso-ssm) l = sso-ssm;
         ss  = (unsigned int) l;
         if (HIFI) wi(ss); else putc(ss,fp);  /* Right channel */
         k++;
      }
      fclose(fp);

      playSound((char*)FNAME);  /* Play the soundscape */
   /* remove("hificode.wav"); */

      k=0;  /* Reset sample count */
   }

   static void *init_ar() {
      Py_Initialize();

      import_array();
      return NUMPY_IMPORT_ARRAY_RETVAL;
   }

   BOOST_PYTHON_MODULE (soundvision_c) {
        //using namespace XM;
        init_ar();

        setup();

        //initialize converters
        to_python_converter<cv::Mat,
                soundvision_c::matToNDArrayBoostConverter>();
        soundvision_c::matFromNDArrayBoostConverter();

        //expose module-level functions
         def("process_frame", process_frame);

    }
}
