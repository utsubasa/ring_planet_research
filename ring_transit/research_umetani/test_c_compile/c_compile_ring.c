
#include <Python.h>
//#include </Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/numpy/ndarraytypes.h>
#include </Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks/Python.framework/Versions/Current/Extras/lib/python/numpy/core/include/numpy/ndarraytypes.h>
#include <math.h>
#include "ring_function_for_python.h"

/*  wrapped cosine function */
static PyObject* ringmodel_getflux(PyObject* self, PyObject* args)
{

    PyArrayObject *time, *par;
    int datanum;

  if (!PyArg_ParseTuple(args, "OOi", &time, &par, &datanum))
    return NULL;

  double *ptime = (double *)time->data;
  double *ppar  = (double *)par->data;
  double *fluxes;
  if ( (fluxes = malloc(sizeof(double)*datanum)) == NULL) {
    printf("Can't allocate memory.\n");
    exit(1);
  }
  get_flux(ptime, ppar, datanum, fluxes);
 int i;
  PyObject *list = PyList_New(0);
  for (i=0; i<datanum; i++)
    PyList_Append(list, Py_BuildValue("d", fluxes[i]));

  free(fluxes);
  return list;

}
#if PY_MAJOR_VERSION >= 3

static PyMethodDef Ringmodelmethods[] = {
  {"getflux", ringmodel_getflux, METH_VARARGS, "return flux values"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "c_compile_ring", "Calculation for ring lightcurves",
    -1,
    Ringmodelmethods
};



PyMODINIT_FUNC
PyInit_c_compile_ring(void)
{
    return PyModule_Create(&cModPyDem);
}


#else

PyMODINIT_FUNC initringmodel_module(void)
{
  (void) Py_InitModule("c_compile_ring", Ringmodelmethods);
}

#endif
