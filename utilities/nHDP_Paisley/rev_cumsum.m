function vec = rev_cumsum(a)

[d1,d2] = size(a);

if d1 > d2
    a = flipud(a);
    vec = cumsum(a);
    vec = flipud(vec);
else
    a = fliplr(a);
    vec = cumsum(a);
    vec = fliplr(vec);
end