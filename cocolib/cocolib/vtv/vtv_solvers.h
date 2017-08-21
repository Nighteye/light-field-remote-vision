#ifndef __MULTIDIM_TV_ROF_SOLVERS_H
#define __MULTIDIM_TV_ROF_SOLVERS_H

#include <gsl/gsl_matrix.h>
#include <vector>

#include "../common/gsl_image.h"


struct multidim_tv_rof_solver_data
{
  // In
  std::vector<gsl_matrix*> _F;
  double _lambda;
  size_t _maxiter;

  // Inout
  std::vector<gsl_matrix*> _U;

  // Original for testing
  std::vector<gsl_matrix*> _O;

  // Out
  double _energy;
  double _energy_smooth;
  double _energy_data;
};

// Map solver to images
multidim_tv_rof_solver_data* multidim_tv_rof_solver_create( gsl_image *U, gsl_image *F );
// Destroy solver
bool multidim_tv_rof_solver_free( multidim_tv_rof_solver_data *data );

// Perform steepest descent for TV-ROF model with spectral norm
// Nonnegativity discretization
bool cpu_multidim_tv_rof_spectral_descent( multidim_tv_rof_solver_data &data );

// Perform primal-dual optimization for TV-ROF model with spectral norm solver
bool cpu_multidim_tv_rof_spectral_pd( multidim_tv_rof_solver_data &data );

// Perform primal-dual optimization for TV-ROF model with Channel-By-Channel
bool cpu_multidim_tv_rof_cbc_pd( multidim_tv_rof_solver_data &data );

// Perform primal-dual optimization for TV-ROF model with DAV solver
bool cpu_multidim_tv_rof_dav_pd( multidim_tv_rof_solver_data &data );

// Perform primal-dual optimization for TV-ROF model with correct convex hull solver
bool cpu_multidim_tv_rof_cvh( multidim_tv_rof_solver_data &data );




#endif
