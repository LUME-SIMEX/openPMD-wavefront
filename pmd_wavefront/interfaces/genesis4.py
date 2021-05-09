import numpy
import h5py

Z0 = np.pi*119.9169832 # V^2/W exactly

def genesis_dfl_from_openpmd_wavefront(h5_meshes, genesis4_filename=None):
    """
   Produces a genesis 2/4 dfl arrray and saves it to genesis 4 field file
   Example:
   genesis4_filename = 'genesis4_field'
   h5 = File('wavefront.h5', 'r')
   meshes = h5['data']['000000']['meshes']
   dfl_x, dfl_y = genesis_dfl_from_openpmd_wavefront(meshes, genesis4_filename)
   
   Parameters
   ----------
   h5_meshes : h5 handle
               open h5 handle to openpmd-wavefront file
   genesis4_filename: str
               genesis 4 field filename
   
   Returns
   -------
   dfl_x, dfl_y : numpy.array
                 3d complex dfl grid with shape (nx, ny, nz)
   
   field values in Genesis4 format
        
    """
        
    # Point to the real and imaginary h5 handles
    E_complex = h5_meshes['electricField']
    # Get attributes
    attrs = dict(E_complex.attrs)
    
    gridsize, gridsize, slicespacing =  E_complex.attrs['gridSpacing']
    refposition = E_complex.attrs['timeOffset']
    wavelength = 12398.425 / E_complex.attrs['photonEnergy'] * 1.0e-10
    
    gridpoints, gridpoints, slicecount = np.array(E_complex['x'].shape)
    params = dict(gridpoints=gridpoints, slicecount=slicecount, gridsize=gridsize, refposition=refposition, slicespacing=slicespacing, wavelength= wavelength)

    factorGenesis = gridsize / np.sqrt(2.0 * Z0) #factor to genesis units
    unitSI = E_complex['x'].attrs['unitSI'] # factor to convert to V/m
    dflfactor = unitSI * factorGenesis # V/m -> Genesis
    
    assert ('x' in E_complex) or ('y' in E_complex), 'missing field components'
    
    # Get arrays
    if 'x' in E_complex:
        
        dfl_x = dflfactor * E_complex['x'][:,:,:]
        
        if genesis4_filename:
            write_dfl_to_genesis4_field(genesis4_filename+'_x.h5',  dflfactor * E_complex['x'][:,:,:], **params)
        
    else:
        dfl_x = None
        
        
    if 'y' in E_complex:
        
        dfl_y = dflfactor * E_complex['y'][:,:,:]
        
        if genesis4_filename:
            write_dfl_to_genesis4_field(genesis4_filename+'_y.h5', dflfactor * E_complex['y'][:,:,:], **params)
            
    else:
        dfl_y = None
        
    
    return dfl_x, dfl_y


def write_dfl_to_genesis4_field(filename, dfl, *, gridpoints, gridsize, refposition, wavelength, slicecount, slicespacing):
    """
    Write dfl array to an open H5 handle in genesis4 format.
    
    filename: str
              Genesis 4 field file name
    dfl: numpy.ndarray
        3d complex dfl grid with shape (nx, ny, nz)
    param: Genesis parameter dict. This routine extracts:
        gridpoints (ncar in v2)
        gridsize (dgrid in v2)
        refposition (ntail in v2)
        wavelength (xlamds in v2)
        slicecount (nslice in v2)
        slicespacing (zsep in v2)
    to write the appropriate metadata.
    
    """
    
    h5 = h5py.File(filename, 'w')
    
    vars = [gridpoints, gridsize, refposition, wavelength, slicecount, slicespacing]
    names = ["gridpoints", "gridsize", "refposition", "wavelength", "slicecount", "slicespacing"]
    types = ['i4', 'f8', 'f8', 'f8', 'i4', 'f8']
    
    for var, name, type in zip(vars, names, types):
        
        dset = h5.create_dataset(name, (1,), dtype=type)
        dset[:] = var
    
    dfl = dfl.reshape((gridpoints**2, slicecount))
    
    for i in range(0, slicecount):
        ind = i + 1
        g = h5.create_group('slice' + str(f'{ind:06}'))
        g['field-real'] = np.real(dfl[:, i]).astype('f8')
        g['field-imag'] = np.imag(dfl[:, i]).astype('f8')    
        
    h5.close()
    
