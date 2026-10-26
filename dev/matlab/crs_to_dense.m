function D = crs_to_dense(rowptr, colidx, values)
	N = length(rowptr) - 1;
	D = zeros(N, N);

	for i=1:N
		b=rowptr(i);
		e=rowptr(i+1);
		for k=b:(e-1)
			j = colidx(k);
			v = values(k);
			D(i, j) = v;
		end
	end