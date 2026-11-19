function [array] = read_array(path, dtype)
	fileID = fopen(path);
	array = fread(fileID, dtype);
	fclose(fileID);
