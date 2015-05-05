function batchwords = GetBatch(fid, nbatch, trainfile)

if fid < 0
    fid = fopen(trainfile);
end

doc_idx = [];
word_idx = [];

line = fgets(fid);
f = 1;
while ischar(line)
    c = textscan(line, '%d');
    c = c{1};
    n = length(c);
    doc_idx = [doc_idx; f * ones(n, 1)];
    word_idx = [word_idx; c];
    if f == nbatch
        break;
    end
    f = f + 1;
    line = fgets(fid);
end

if f ~= nbatch && ~ischar(line)
    fclose(fid);
    fid = fopen(trainfile);
end

while 1
    if f == nbatch
        break;
    end
    line = fgets(fid);
    assert(ischar(line));
    c = textscan(line, '%d');
    c = c{1};
    n = length(c);
    doc_idx = [doc_idx; f * ones(n, 1)];
    word_idx = [word_idx; c];
    if f == nbatch
        break;
    end
    f = f + 1;
end

word_idx = word_idx + 1;

batchwords = sparse(double(word_idx), double(doc_idx), ones(length(doc_idx), 1), 7702, nbatch);
