function [rowptr, colidx, values] = read_crs(folder)
	path_rowptr = [folder, '/rowptr.raw'];
	path_colidx = [folder, '/colidx.raw'];
	path_values = [folder, '/values.raw'];

	rowptr = read_array(path_rowptr, 'int32');
	colidx = read_array(path_colidx, 'int32');
	values = read_array(path_values, 'double');

	rowptr = rowptr + 1;
	colidx = colidx + 1;