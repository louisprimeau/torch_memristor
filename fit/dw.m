function out = dw(w, v, p)
	 f = @(x, a, b) (heaviside(x-a) - heaviside(x-b));
	 if v > p(8)
	    out = p(4) * ((v / p(8) - 1)^p(6)) * f(w, p(1), p(2));
	 else if v < p(9)
	    out = p(5) * (v / p(9) - 1)^p(7) * f(w, p(1), p(2));
	 else
	    out = 0;
	 end
	 
end
