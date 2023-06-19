function [rowptr, colidx, values] = read_block_crs(block_rows, block_cols, folder)
	path_rowptr = [folder, '/rowptr.raw'];
	path_colidx = [folder, '/colidx.raw'];
	rowptr = read_array(path_rowptr, 'int');
	colidx = read_array(path_colidx, 'int');
	path_values = [folder, '/values.%d.raw']; 
	values = cell(block_rows, block_cols);
	for i=0:(block_rows-1)
		for j=0:(block_cols-1)
			path_values = sprintf(path_values, (i*block_cols + j));
			values{i+1, j+1} = read_array(path_values, 'double');
		end
	end