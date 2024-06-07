// a,b lengths of incoming rectangle interpreted as polar coordinates, rescaled to [0,2pi]xR and mapped to Cartesian system
void polar_warp(f32* out, f32* in, u32& n_out, u32 n_in, f32 a, f32 b, f32 R)
{
	for (u32 u{}; u < n_out && u < n_in; u++)
	{
		f32 phi = tau*in[2 * u + 0]/a;
		f32 r   = R  *std::sqrt(in[2 * u + 1]/b);
		out[2 * u + 0] = r*std::cos(phi);
		out[2 * u + 1] = r*std::sin(phi);
	}
	n_out = n_in;
}

void concentric_warp(f32* out, f32* in, u32& n_out, u32 n_in, f32 a, f32 b, f32 R)
{
	for (u32 u{}; u < n_out && u < n_in; u++)
	{
		f32 x = 2*in[2 * u + 0]/a-1;
		f32 y = 2*in[2 * u + 1]/b-1;
		f32 r, phi;
		if (x == 0 && y == 0)
		{
			r   = 0;
			phi = 0;
		}
		else if (std::abs(x) > std::abs(y)) 
		{
			r   = x*R;
			phi = pi_f/4 * (y / x);

		} 
		else 
		{
			r  = y*R;
			phi= pi_f/2 - pi_f/4 * (x / y);
		}
		out[2 * u + 0] = r*std::cos(phi);
		out[2 * u + 1] = r*std::sin(phi);
	}
	n_out = n_in;
}

void rejection_warp(f32* out, f32* in, u32& n_out, u32 n_in, f32 a, f32 b, f32 R)
{
	u32 m{};
	for (u32 u{}; u < n_in && m<n_out; u++)
	{
		f32 x = 2 * in[2 * u + 0] / a - 1;
		f32 y = 2 * in[2 * u + 1] / b - 1;

		f32 r2 = x * x + y * y;
		if (r2 > 1.)
			continue;

		out[2 * m + 0] = R *  x;
		out[2 * m + 1] = R *  y;
		m++;
	}
	n_out = m;
}

void adoption_warp(f32* out, f32* in, u32& n_out, u32 n_in, f32 a, f32 b, f32 R)
{
	f32 s2 = std::sqrt(2.f);
	u32 m{};
	for (u32 u{}; u < n_in && m<n_out; u++)
	{
		f32 x = 2 * in[2 * u + 0] / a - 1;
		f32 y = 2 * in[2 * u + 1] / b - 1;

		out[2 * m + 0] = (R * s2 / 2) * x;
		out[2 * m + 1] = (R * s2 / 2) * y;
		m++;

		f32 r2 = x * x + y * y;
		f32 t = r2 + 2.f;
		f32 x_cache = x;
		f32 y_cache = y;
		if (t <= 4 * x)
		{
			x_cache -= 2;
		}
		else if (t <= -4 * x)
		{
			x_cache += 2;
		}
		else if (t <= 4 * y)
		{
			y_cache -= 2;
		}
		else if (t <= -4 * y)
		{
			y_cache += 2;
		}
		else
			continue;
		if (m < n_out)
		{
			out[2 * m + 0] = (R * s2 / 2) * x_cache;
			out[2 * m + 1] = (R * s2 / 2) * y_cache;
			m++;
		}
	}
	n_out = m;
}
