
import numpy as np

from array import array
from copy import deepcopy
import struct

Z0 = np.pi*119.9169832 # V^2/W exactly




def srwl_uti_array_alloc(_type, _n):
    """
    Copy of SRW's routing for preparation of SRWLWrf Class
    """
    nPartMax = 10000000 #to tune
    if(_n <= nPartMax): return array(_type, [0]*_n)
        #resAr = array(_type, [0]*_n)
        #print('Array requested:', _n, 'Allocated:', len(resAr))
        #return resAr

    nEqualParts = int(_n/nPartMax)
    nResid = int(_n - nEqualParts*nPartMax)
    resAr = array(_type, [0]*nPartMax)
    if(nEqualParts > 1):
        auxAr = deepcopy(resAr)
        for i in range(nEqualParts - 1): resAr.extend(auxAr)
    if(nResid > 0):
        auxAr = array(_type, [0]*nResid)
        resAr.extend(auxAr)

    #print('Array requested:', _n, 'Allocated:', len(resAr))
    return resAr


def np_complex_to_srw_array(cdat, dtype=np.dtype('f')):
    """
    Convert the 3D complex array with [x,y,z] to SRW's flat array
    """
    dat = np.moveaxis(cdat, [0,1,2], [1,0,2]).flatten() # [x,y,z] to # [y,x,z]
    x_np = np.ravel([np.real(dat), np.imag(dat)], order = 'F').astype(dtype)
    return array(x_np.dtype.char, x_np.data.tobytes())


def srw_wfr_from_openpmd_wavefront(h5_meshes, dtype=np.dtype('f'), iz_start=None, iz_end=None, iz_step=None, SRWLWfr_class=None):
    """
    Forms the essential input for SRW's wfr class from an open handle to openPMD-wavefront meshes.

    Optionally iz_start, iz_end, and iz_step can be given which appropriately sample the full data in the z dimension. 
    
    SRWLWfr_class should be the SRWLWfr class. If this is not given, then the raw inputs needed will be returned:
        arEx, arEy, kwargs,  wrf_attrs
    And then the wavefront can be constructed with:
        wfr = SRWLWfr_class(arEx, arEy, **kwargs)
        wfr.__dict__.update(wrf_attrs)        
        
    Note that the units in an openPMD-wavefront electric field E are V/m, and SRW's field F_SRW are sqrt(W/mm^2):
        E = F_SRW * sqrt(2*Z_0) * 1000 mm/m
        
    
    Example with SRW:
        h5 = File('wavefront.h5', 'r')
        meshes = h5['data']['000000']['meshes']
        
        from ?? import SRWLWfr
        
        wrf = srw_wfr_from_openpmd_wavefront(meshes)
        
    """
    # Point to the real and imaginary h5 handles
    E_complex = h5_meshes['electricField']
    # Get attributes
    attrs = dict(E_complex.attrs)
    
    # Make slice object
    zslice = slice(iz_start, iz_end, iz_step)
    
    assert ('x' in E_complex) or ('y' in E_complex)
    
    # Get arrays
    if 'x' in E_complex:
        shape = np.array(E_complex['x'].shape)
        unitSI = E_complex['x'].attrs['unitSI'] # factor to convert to V/m
        srwfactor = unitSI / np.sqrt(2*Z0) / 1000 # V/m -> sqrt(W/mm^2)        
        arEx = np_complex_to_srw_array(E_complex['x'][:,:,zslice]* srwfactor , dtype=dtype) 
    else:
        arEx = None
        
    if 'y' in E_complex:
        shape = np.array(E_complex['y'].shape)
        unitSI = E_complex['y'].attrs['unitSI'] # factor to convert to V/m
        srwfactor = unitSI / np.sqrt(2*Z0) / 1000 # V/m -> sqrt(W/mm^2)        
        arEy = np_complex_to_srw_array(E_complex['y'][:,:,zslice]* srwfactor , dtype=dtype) 
    else:
        arEy = None
        
    # Make empty arrays for the missing ones. 
    if arEx is None: arEx = srwl_uti_array_alloc('f', len(arEy))
    if arEy is None: arEy = srwl_uti_array_alloc('f', len(arEx))
    
    c_light = 299792458. # m/s
    #freq = attrs['frequency']*attrs['frequencyUnitSI']
    
    # Photon energy 
    photon_energy_eV = attrs['photonEnergy']*attrs['photonEnergyUnitSI']/1.602176634e-19 # J -> eV
    
    # info about the full grid
    delta  = attrs['gridSpacing']
    mins = attrs['gridGlobalOffset']
    maxs = (shape-1)*delta + mins
    
    # Handle z slicing
    dz = delta[2]
  #  print('shape', shape)
  #  print('dz:', dz)
    nz = shape[2]    

    # calculate sub-sampled dimensions
    # Handle negative indices
    
    if iz_start is None:
        iz_start = 0
    if iz_end is None:
        iz_end = nz 
    if iz_step is None:
        iz_step = 1
    
    if iz_start < 0:
        iz_start = nz + iz_start
    if iz_end < 0:
        iz_end = nz + iz_end
    
    
    
    # Actual end iz if there is a step
    iz_end = iz_step*((iz_end-iz_start)//iz_step) + iz_start        
    nz = (iz_end - iz_start)//iz_step
    
    #print('iz start, end, nz', iz_start, iz_end, nz)
    #print('z ptp original:', (maxs[2]-mins[2]), iz_end-iz_start)
    
    zmin = iz_start*dz + mins[2]
    zmax = (iz_end-1)*dz   + mins[2]

    
   # print('zmin, mins[2]', zmin, mins[2])
    #print('zmax, maxs[2]', zmax, maxs[2])
    
    # Form kwargs for SRWLWfr init
    kwargs=dict(_typeE    = dtype.char,
                  _eStart = 0, #zmin/c_light,
                  _eFin   = (zmax-zmin)/c_light,
                  _ne     = nz,
                  _xStart =mins[0],
                  _xFin   =maxs[0],
                  _nx     =shape[0],
                  _yStart =mins[1],
                  _yFin   =maxs[1],
                  _ny     =shape[1],
                  _zStart=0) #??? what is this
    
    # wfr. items
    presFT = 1 #presentation/domain: 0- frequency (photon energy), 1- time
    #wfr.unitElFld = 2 #electric field units are sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)
    wrf_attrs = dict(avgPhotEn = photon_energy_eV, presFT = presFT, unitElFld = 2)
    
    
    # 
    if SRWLWfr_class is None:
        return arEx, arEy, kwargs,  wrf_attrs

    # This forms the wavefront object
    wfr = SRWLWfr_class(arEx, arEy, **kwargs)
    wfr.__dict__.update(wrf_attrs)
    
    return wfr