f32 gaussian(f32 x, f32 y)
{
	return std::exp(-5.*(x * x + y * y));
}

f32 quatered_disk(f32 x, f32 y)
{
	return x*y > 0 ? 1.f : 0.f;
}

f32 square_in_disk(f32 x, f32 y)
{
	return std::max(std::abs(x),std::abs(y)) < std::sqrt(2)/2 ? 1.f : 0.f;
}

f32 square2_in_disk(f32 x, f32 y)
{
	return std::abs(x)+std::abs(y) < 1 ? 1.f : 0.f;
}

f32 concentric(f32 x, f32 y)
{
	f32 r = std::sqrt(x * x + y * y);
	f32 s = x > 0;
	return (u32)(10*r) %2 == 0 ? s : 1-s;
}

f32 spiral(f32 x, f32 y)
{
	f32 theta = std::atan2(y,x);
	if (theta < 0) theta += tau;
	f32 r = std::sqrt(x * x + y * y);
	return r * (theta / tau);
}


f32 checkerboard(f32 x, f32 y)
{

	u32 n = 16;
	u32 a = (x+y + 2) / 4 * n;
	u32 b = (x-y + 2) / 4 * n;
	return  (a+b)%2;
}

f32 bilinear(f32 x, f32 y)
{
	return (x + 1) / 2 * (y + 1) / 2 * 0.3 ;
}

using func_2d = f32(*)(f32, f32);

f32 function_integral(func_2d f)
{
	f64 sum{};
	u32 n{};
	u32 m = 10000;
	for (u32 u{}; u<m; u++) for (u32 v{}; v<m; v++)
	{
		f32 x = 2*(f32)u/m - 1.f;
		f32 y = 2*(f32)v/m - 1.f;
		if (x*x + y*y <= 1.f)
		{
			sum += f(x, y);
			n++;
		}
	}
	return sum / n;
}

f32 checkerboard_integral()
{
	return function_integral(checkerboard);
}

f32 bilinear_integral()
{
	return function_integral(bilinear);
}

stb_image head_img{};
stb_image drum_img{};

f32 image_integral(stb_image img)
{
	f64 sum{};
	u32 n{};
	for (u32 x{}; x<img.w; x++)
	for (u32 y{}; y<img.h; y++)
	{
		f64 xf = 2*(1. / img.w * x)-1;
		f64 yf = 2*(1. / img.h * y)-1;
		f64 r2 = xf * xf + yf * yf;
		if (r2 > 1.) continue;

		u32 u = (x * img.h + y) * 4;
		u32 r = img.data[u];
		u32 g = img.data[u+1];
		u32 b = img.data[u+2];
	
		n++;
		sum += (1.f / 255 * (r + g + b) / 3);
	}

	return sum / n;
}

f32 image_val(stb_image img, f32 x, f32 y)
{
	u32 xi = std::min( (u32)((x + 1.f) / 2 * img.w), (u32)img.w-1);
	u32 yi = std::min( (u32)((y + 1.f) / 2 * img.h), (u32)img.h-1);

	u32 u = (xi * img.h + yi) * 4;

	u32 r = img.data[u];
	u32 g = img.data[u+1];
	u32 b = img.data[u+2];
	return (1.f / 255 * (r + g + b) / 3);
}

f32 head(f32 x, f32 y)
{
	return image_val(head_img, x, y);
}

f32 head_integral()
{
	return image_integral(head_img);
}

f32 drum(f32 x, f32 y)
{
	return image_val(drum_img, x, y);
}

f32 drum_integral()
{
	return image_integral(drum_img);
}

// Normalize the measure over the unit disk to one.
const f32 gaussian_integral   = 2 * 0.1 * (1.f - std::exp(-5.f)); 
const f32 quatered_disk_integral  = 1./2;
const f32 concentric_integral = 1./2;
const f32 spiral_integral     = 1./3;
const f32 square_integral     = 2./pi_f;
const f32 square2_integral     = 2./pi_f;