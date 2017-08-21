TEMPLATE	 = lib
VERSION          = 0.0.1
CONFIG		+= qt debug staticlib

HEADERS		= defs.h\
                  \
                  common/gsl_image.h\
                  common/gsl_image_statistics.h\
                  common/gsl_matrix_helper.h\
                  common/gsl_matrix_derivatives.h\
                  common/gsl_matrix_convolutions.h\
                  common/parse_config.h\
                  common/color_spaces.h\
                  common/histogram.h\
                  common/debug.h\
                  common/profiler.h\
                  common/hdf5_tools.h\
                  common/menger_curvature.h\
                  \
                  cocolib++/compute_api/compute_buffer.h\
                  cocolib++/compute_api/compute_engine.h\
                  cocolib++/compute_api/compute_grid.h\
                  cocolib++/compute_api/compute_array.h\
                  cocolib++/compute_api/reprojections.h\
                  cocolib++/compute_api/kernels_algebra.h\
                  cocolib++/compute_api/kernels_reprojections.h\
                  cocolib++/compute_api/kernels_vtv.h\
                  cocolib++/compute_api/kernels_multilabel.h\
                  \
                  cocolib++/models/variational_model.h\
                  cocolib++/models/inverse_problem.h\
                  cocolib++/models/multilabel_problem.h\
                  \
                  cocolib++/regularizers/regularizer.h\
                  cocolib++/regularizers/vtv_s.h\
                  cocolib++/regularizers/vtv_f.h\
                  cocolib++/regularizers/vtv_j.h\
                  cocolib++/regularizers/tgv_2.h\
                  cocolib++/regularizers/multilabel.h\
                  cocolib++/regularizers/multilabel_potts.h\
                  cocolib++/regularizers/multilabel_decision.h\
                  \
                  cocolib++/data_terms/data_term.h\
                  cocolib++/data_terms/rof.h\
                  cocolib++/data_terms/denoising.h\
                  cocolib++/data_terms/deconvolution.h\
                  cocolib++/data_terms/multilabel_linear_cost.h\
                  \
                  cocolib++/solvers/solver.h\
                  cocolib++/solvers/solver_chambolle_pock.h\
                  cocolib++/solvers/stopping_criterion.h
                  

SOURCES         = common/gsl_image.cpp\
                  common/gsl_image_statistics.cpp\
                  common/gsl_matrix_helper.cpp\
                  common/gsl_matrix_derivatives.cpp\
                  common/gsl_matrix_convolutions.cpp\
                  common/parse_config.cpp\
                  common/color_spaces.cpp\
                  common/histogram.cpp\
                  common/debug.cpp\
                  common/profiler.cpp\
                  common/menger_curvature.cpp\
                  common/hdf5_tools.cpp\
                  \
                  cocolib++/compute_api/compute_array.cpp\
                  cocolib++/compute_api/reprojections.cpp\
                  cocolib++/compute_api/convolutions_api.cpp\
                  \
                  cocolib++/regularizers/regularizer.cpp\
                  cocolib++/regularizers/vtv_s.cpp\
                  cocolib++/regularizers/vtv_f.cpp\
                  cocolib++/regularizers/vtv_j.cpp\
                  cocolib++/regularizers/tgv_2.cpp\
                  cocolib++/regularizers/multilabel.cpp\
                  cocolib++/regularizers/multilabel_potts.cpp\
                  cocolib++/regularizers/multilabel_decision.cpp\
                  \
                  cocolib++/data_terms/data_term.cpp\
                  cocolib++/data_terms/rof.cpp\
                  cocolib++/data_terms/denoising.cpp\
                  cocolib++/data_terms/deconvolution.cpp\
                  cocolib++/data_terms/multilabel_linear_cost.cpp\
                  \
                  cocolib++/models/variational_model.cpp\
                  cocolib++/models/inverse_problem.cpp\
                  cocolib++/models/multilabel_problem.cpp\
                  \
                  cocolib++/solvers/solver.cpp\
                  cocolib++/solvers/solver_chambolle_pock.cpp\
                  cocolib++/solvers/stopping_criterion.cpp\
                  \
                  cocolib++/compute_api_implementation_opencl/compute_buffer.cpp\
                  cocolib++/compute_api_implementation_opencl/compute_engine.cpp\
                  cocolib++/compute_api_implementation_opencl/compute_grid.cpp\
                  cocolib++/compute_api_implementation_opencl/kernels_vtv.cpp\
                  cocolib++/compute_api_implementation_opencl/kernels_reprojections.cpp\
                  cocolib++/compute_api_implementation_opencl/kernels_algebra.cpp\
                  cocolib++/compute_api_implementation_opencl/convolutions.cpp
		          
                  
DISTFILES = README\
            relink\
            \
            common/makefile\
            cuda/makefile\
            tv/makefile\
            vtv/makefile\
            tc/makefile\


TARGET		= cocolib


include("../extra_qmake_flags.pri")
include("../extra_libs.pri")

QMAKE_CXXFLAGS += -I.

linux-g++: LIBS += -lglut -lGL
macx-g++: LIBS+= -framework GLUT

OBJECTS_DIR = ./obj
