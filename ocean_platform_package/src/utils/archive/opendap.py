# This module contains adapters and a facade for the different python netcdf libraries.

# $Id: opendap.py 4658 2011-06-13 15:41:23Z boer_g $
# $Date: 2011-06-13 17:41:23 +0200 (ma, 13 jun 2011) $
# $Author: boer_g $
# $Revision: 4658 $
# $HeadURL: https://repos.deltares.nl/repos/OpenEarthTools/trunk/python/io/opendap/opendap.py $
# $Keywords: $

import warnings

['close', 'cmptypes', 'createCompoundType', 'createDimension', 'createGroup', 'createVLType', 'createVariable',
 'delncattr', 'dimensions', 'file_format', 'getncattr', 'groups', 'maskanscale', 'ncattrs', 'parent', 'path',
 'renameDimension', 'renameVariable', 'set_fill_off', 'set_fill_on', 'setncattr', 'sync', 'variables', 'vltypes']


def pydaptonetCDF4(dataset):
	"""make a pydap dataset look and quack like a netcdf4 dataset
	>>> import pydap.client
	>>> url  = 'http://opendap.deltares.nl:8080/opendap/rijkswaterstaat/jarkus/profiles/transect.nc'
	>>> ds   = pydap.client.open_url(url)
	>>> ncds = pydaptonetCDF4(ds)
	>>> type(ncds)
	<class 'pydap.model.DatasetType'>

	You should now be able to access the dataset in a netCDF4 way (with variables)
	>>> ncds.variables['x'] is ds['x']['x']
	True
	"""
	import pydap.model
	assert isinstance(dataset, pydap.model.DatasetType)
	# in pydap the dataset itself is a dict, in netCDF4 it has a variables dict
	# let's add the variables as well
	dataset.variables = {}
	for variable in dataset.keys():
		if isinstance(dataset[variable], pydap.model.GridType):
			# the pydap client returns grids for arrays with coordinates.
			#
			dataset.variables[variable] = dataset[variable][variable]
			dataset.variables[variable].attributes.update(dataset[variable].attributes)
		else:
			dataset.variables[variable] = dataset[variable]
	for key, value in dataset.attributes['NC_GLOBAL'].items():
		if key not in dataset:
			# I think the __setitem__ might be overwritten, so we'll do it like this
			setattr(dataset, key, value)
		else:
			warnings.warn('Could not set %s to %s because it already exists as a variable' % (key, value))
	return dataset


# TO DO
# deal with fill_values: netCDF deals with them automaticaly
# by inserting into a masked arrray, whereas pydap does not.
# numpy.ma.MaskedArray or does netCDF4 doe that based on presence of fillvalue att

# TO DO
# perhaps create a netCDF4topydap adapter also

def opendap(url):
	"""return the dataset looking like a netCDF4 object
	>>> url = 'http://opendap.deltares.nl:8080/opendap/rijkswaterstaat/jarkus/profiles/transect.nc'
	>>> ds = opendap(url)
	>>> ds.variables.has_key('x')
	True
	>>> ds.title
	'Jarkus Data'

	"""
	try:
		import netCDF4
		dataset = netCDF4.Dataset(url)
	# Either netcdf is not found, or it cannot read url's
	except (ImportError, RuntimeError):
		import pydap.client
		dataset = pydaptonetCDF4(pydap.client.open_url(url))
	return dataset


if __name__ == "__main__":
	import doctest

	doctest.testmod()