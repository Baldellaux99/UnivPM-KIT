function write_full_matrix(G,filename)
[fileid,message]=fopen(filename,'wt');
if (fileid==-1)
    error(message)
end
dlmwrite(filename, G,  '-append', 'delimiter', ' ', 'precision','%d');