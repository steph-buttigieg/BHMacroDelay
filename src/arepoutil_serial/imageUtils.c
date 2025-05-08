#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double getWk(double u) {
  if(u < 0.5)
    return (2.546479089470 + 15.278874536822 * (u - 1) * u * u);
  else
    return 5.092958178941 * (1.0 - u) * (1.0 - u) * (1.0 - u);
}

int getArrays(int Xpixels, int Ypixels, double weight[], double quant[], double xa[], double ya[], double za[], double ha[], double width, double height, double halfDepth, int NumPart, double weightArray[], double quantArray[], bool calcQuant)
{
  int i, j, n;
  int dx, dy, nx, ny;
  double h, r, u, wk;
  double x0, y0;
  double r2, h2;
  double sum, hmin, hmax, x, y, xx, yy, xxx, yyy;
  double pixelsizeX, pixelsizeY;

  pixelsizeX = width / Xpixels;
  pixelsizeY = height / Ypixels;

  if(pixelsizeX < pixelsizeY)
    hmin = 1.001 * pixelsizeX / 1;
  else
    hmin = 1.001 * pixelsizeY / 1;


  if(pixelsizeX < pixelsizeY)
    hmax = 64 * pixelsizeX;
  else
    hmax = 64 * pixelsizeY;

  for(n = 0; n < NumPart; n++) {
    if(za[n] < -halfDepth || za[n] > halfDepth)
      continue;

    x0 = xa[n];
    y0 = ya[n];
    h = ha[n];

    if(h < hmin)
      h = hmin;

    if(h > hmax)
      h = hmax;

    if(x0 + h < 0 || x0 - h >  width
      || y0 + h < 0 || y0 - h > height)
      continue;

    h2 = h * h;

    nx = h / pixelsizeX + 1;
    ny = h / pixelsizeY + 1;

    /* x,y central pixel of region covered by the particle on the mesh */

    x = (floor(x0 / pixelsizeX) + 0.5) * pixelsizeX;
    y = (floor(y0 / pixelsizeY) + 0.5) * pixelsizeY;

    /* determine kernel normalizaton */
    sum = 0;

    for(dx = -nx; dx <= nx; dx++)
      for(dy = -ny; dy <= ny; dy++) {
        xx = x + dx * pixelsizeX - x0;
        yy = y + dy * pixelsizeY - y0;
        r2 = xx * xx + yy * yy;

        if(r2 < h2) {
          r = sqrt(r2);
          u = r / h;
          wk = getWk(u);
          sum += wk;
        }
      }

    if(sum < 1.0e-10)
      continue;

    for(dx = -nx; dx <= nx; dx++)
      for(dy = -ny; dy <= ny; dy++) {
        xxx = x + dx * pixelsizeX;
        yyy = y + dy * pixelsizeY;

        if(xxx >= 0 && yyy >= 0) {
          i = xxx / pixelsizeX;
          j = yyy / pixelsizeY;

          if(i >= 0 && i < Xpixels)
            if(j >= 0 && j < Ypixels) {
              xx = x + dx * pixelsizeX - x0;
              yy = y + dy * pixelsizeY - y0;
              r2 = xx * xx + yy * yy;

              if(r2 < h2) {
                r = sqrt(r2);
                u = r / h;
                wk = getWk(u);
                weightArray[i * Ypixels + j] += weight[n] * wk / sum;
                if(calcQuant == true)
                  quantArray[i * Ypixels + j] += weight[n] * quant[n] * wk / sum;
              }
            }
        }
      }
  }
  return 0;
}

int getMassArray(int Xpixels, int Ypixels, double mass[], double xa[], double ya[], double za[], double ha[], double width, double height, double halfDepth, int NumPart, double weightArray[]) {
  double quant[0];
  double quantArray[0];
  getArrays(Xpixels, Ypixels, mass, quant, xa, ya, za, ha, width, height, halfDepth, NumPart, weightArray, quantArray, false);
  return 0;
}

int getWeightedQuantityArray(int Xpixels, int Ypixels, double weight[], double quant[], double xa[], double ya[], double za[], double ha[], double width, double height, double halfDepth, int NumPart, double weightArray[], double quantArray[]) {
  getArrays(Xpixels, Ypixels, weight, quant, xa, ya, za, ha, width, height, halfDepth, NumPart, weightArray, quantArray, true);
    return 0;
}
