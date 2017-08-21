/** \file visualization.cpp

    Functions to visualize certain stuff (vector fields etc.)
    
    Copyright (C) 2013 Bastian Goldluecke.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
   
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <qimage.h>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <float.h>

#include <QPixmap>
#include <QPainter>

#include "visualization.h"
#include "gsl_image.h"

#include "debug.h"

using namespace std;

// Draw outline of level set using marching squares algorithm
bool coco::draw_vector_field_to_image( gsl_matrix *dx, gsl_matrix *dy,
				       double scale,
				       QImage &I,
				       int xbase, int xstep,
				       int ybase, int ystep,
				       QRgb colorBase, QRgb colorArrow,
				       double baseCircleRadius, double lineWidth )
{
#if QT_VERSION > 0x40000
  QPixmap pm = QPixmap::fromImage( I );
#else
  QPixmap pm( I );
#endif
  QPainter pt(&pm);
  QPen pen_arrows( colorArrow );
  pen_arrows.setWidth( lineWidth );
  QBrush brush( colorBase );
  pt.setBrush( brush );
  //double maxlen = sqrt( xstep*xstep + ystep*ystep );
  double maxlen = min( xstep, ystep );//sqrt( xstep*xstep + ystep*ystep );

  // Trace image
  xbase = max(0,xbase);
  ybase = max(0,ybase);
  int W = I.width();
  int H = I.height();
  for ( int x=xbase; x<W; x += xstep ) {
    for ( int y=ybase; y<H; y += ystep ) {
      TRACE9( "x " << x << " / " << W << endl );
      TRACE9( "y " << y << " / " << H << endl );
      double dxv = scale * gsl_matrix_get( dx, y,x );
      double dyv = scale * gsl_matrix_get( dy, y,x );
      TRACE9( "v " << dxv << " " << dyv << endl );
      double L = hypot( dxv, dyv );
      if ( L>maxlen ) {
	dxv  = dxv / L * maxlen;
	dyv  = dyv / L * maxlen;
      }

      // Draw the arrow
      pt.setPen( pen_arrows );
      pt.drawLine( x,y, int(x+dxv), int(y+dyv) );
      
      // Plot the base pixel
      pt.setPen( colorBase );
      pt.drawEllipse( x - baseCircleRadius, y - baseCircleRadius, baseCircleRadius*2+1, baseCircleRadius*2+1 );
    }
  }

#if QT_VERSION > 0x40000
  I = pm.toImage();
#else 
  I = pm.convertToImage();
#endif

  return true;
}

