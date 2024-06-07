// pbrt 4th. Section 8.6
f32 radical_inverse(u32 b, u32 a)
{
	f32 inv_base = 1.f / b;
	f32 inv_base_mult = 1;
	u32 c{};
	while (a)
	{
		u32 next = a / b;
		u32 digit = a - next * b;
		c = c * b + digit;
		inv_base_mult *= inv_base;
		a = next;
	}
	return std::min(c * inv_base_mult,1.f);

}

#include "sobolmatrices.cpp"
// pbrt 4th. Section 8.7
u32 multiply_generator(u64 a, u32 dim) 
{
    u32 v = 0;
    for (u32 i = dim * SobolMatrixSize; a != 0; ++i, a >>= 1)
        if (a & 1)
            v ^= sobol_matrices32[i];
    return v;
}

//pbrt4 
int Log2Int(u32 v) 
{
    unsigned long lz = 0;
    if (_BitScanReverse(&lz, v))
        return lz;
    return 0;
}

//pbrt4 
constexpr u32 RoundUpPow2(u32 v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// pbrt4 hash.h
uint64_t MixBits(uint64_t v) {
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}
// pbrt4 math.h
int PermutationElement(uint32_t i, uint32_t l, uint32_t p) {
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

const f32 OneMinusEpsilon = f32(0x1.fffffep-1);

//pbrt4 
f32 OwenScrambledRadicalInverse(u32 base, uint64_t a, uint32_t hash) 
{
    // We have to stop once reversedDigits is >= limit since otherwise the
    // next digit of |a| may cause reversedDigits to overflow.
    uint64_t limit = ~0ull / base - base;
    f32 invBase = 1.f / base, invBaseM = 1;
    uint64_t reversedDigits = 0;
    int digitIndex = 0;
    while (1 - invBaseM < 1 && reversedDigits < limit) {
        // Compute Owen-scrambled digit for _digitIndex_
        uint64_t next = a / base;
        int digitValue = a - next * base;
        uint32_t digitHash = MixBits(hash ^ reversedDigits);
        digitValue = PermutationElement(digitValue, base, digitHash);
        reversedDigits = reversedDigits * base + digitValue;
        invBaseM *= invBase;
        ++digitIndex;
        a = next;
    }
    return std::min(invBaseM * reversedDigits, OneMinusEpsilon);
}

// pbrt 4th. Section 8.7
u64 sobol_index(u32 m, u64 frame, vec2_u32 p)
{
    if (m == 0)
        return frame;

    const u32 m2 = m << 1;
    u64 index = uint64_t(frame) << m2;

    u64 delta = 0;
    for (int c = 0; frame; frame >>= 1, ++c)
        if (frame & 1)  // Add flipped column m + c + 1.
            delta ^= VdCSobolMatrices[m - 1][c];

    // flipped b
    u64 b = ((u64)(p.x) << m) | p.y ^ delta;

    for (int c = 0; b; b >>= 1, ++c)
        if (b & 1)  // Add column 2 * m - c.
            index ^= VdCSobolMatricesInv[m - 1][c];

    return index;
} 


// pbrt 4th. Section B.2.7
u32 reverse_bits(u32 n) 
{
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
}

points grid(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			vertices[2 * (i + j * m)    ] = a * i / m + a/m/2;
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n/2;
		}
	return points{vertices, n_vertices};
}

points halton(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * radical_inverse(2, u);
			vertices[2 * u + 1] = b * radical_inverse(3, u);
		}
	return points{vertices, n_vertices};
}

points hammersley(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * u / n_vertices;
			vertices[2 * u + 1] = b * radical_inverse(2, u);
		}
	return points{vertices, n_vertices};
}

points sobol(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	u32 dim{0};
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			u64 si = sobol_index(Log2Int(RoundUpPow2(m)), 0, { i,j });
			vertices[2 * u]		= a * 0x1p-32f * multiply_generator(si, dim);
			vertices[2 * u + 1] = b * 0x1p-32f * multiply_generator(si, dim+1);
		}
	return points{vertices, n_vertices};
}

points jitter(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			auto ran = pcg2d(vec2_u32{ i,j });
			
			vertices[2 * (i + j * m)    ] = a * i / m + a/m*ran.x*0x1p-32f;
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n*ran.y*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points random(u32 m, u32 n, f32 a, f32 b, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			auto ran = pcg2d(vec2_u32{ i,j });
			
			vertices[2 * (i + j * m)    ] = a*ran.x*0x1p-32f;
			vertices[2 * (i + j * m) + 1] = b*ran.y*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points jitter(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++) for (u32 i{}; i < m; i++)
		{
			ran = pcg(ran);
			vertices[2 * (i + j * m)    ] = a * i / m + a/m*ran*0x1p-32f;
			ran = pcg(ran);
			vertices[2 * (i + j * m) + 1] = b * j / n + b/n*ran*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

points random(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	
	for (u32 j{}; j < n; j++) for (u32 i{}; i < m; i++)
		{
			ran = pcg(ran);
			vertices[2 * (i + j * m)    ] = a*ran*0x1p-32f;
			ran = pcg(ran);
			vertices[2 * (i + j * m) + 1] = b*ran*0x1p-32f;
		}
	return points{vertices, n_vertices};
}

// pbrt 4th 8.7.2 
u32 permute_scramble(u32 u, u32 p)
{
    return p^u;
};

// pbrt 4th 8.7.2 
u32 fast_owen_scramble(u32 v, u32 seed)
{
	v = reverse_bits(v);
	v ^= v * 0x3d20adea;
	v += seed;
	v *= (seed >> 16) | 1;
	v ^= v * 0x05526c56;
	v ^= v * 0x53a22864;
	return reverse_bits(v);
}


points sobol_scrambled(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	u32 dim{0};
	u32 r1 = pcg(ran);
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			u64 si = sobol_index(Log2Int(RoundUpPow2(m)), 0, { i,j });
			vertices[2 * u]		= a * 0x1p-32f * fast_owen_scramble(multiply_generator(si, dim), r1);
			vertices[2 * u + 1] = b * 0x1p-32f * fast_owen_scramble(multiply_generator(si, dim+1), r1);
		}
	return points{vertices, n_vertices};
}

points halton_scrambled(u32 m, u32 n, f32 a, f32 b,u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	u32 r1 = pcg(ran);
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * OwenScrambledRadicalInverse(2, u,r1);
			vertices[2 * u + 1] = b * OwenScrambledRadicalInverse(3, u,r1);
		}
	return points{vertices, n_vertices};
}

points hammersley_scrambled(u32 m, u32 n, f32 a, f32 b, u32 ran, f32 *vertices)
{
	u32 n_vertices = m * n;
	u32 r1 = pcg(ran);
	for (u32 j{}; j < n; j++)
		for (u32 i{}; i < m; i++)
		{
			u32 u = i + j * m;
			vertices[2 * u    ] = a * u / n_vertices;
			vertices[2 * u + 1] = b * OwenScrambledRadicalInverse(2, u, r1);
		}
	return points{vertices, n_vertices};
}
using scramble_func = u32(*)(u32, u32);
using pts_func = points(*)(u32 m, u32 n, f32 a, f32 b, f32* vertices);

struct scramble_functor 
{
    pts_func sampler;
    scramble_func scrambler;
    scramble_functor(pts_func sampler, scramble_func scrambler) : sampler(sampler) , scrambler(scrambler) {}

    points operator()(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices) const 
	{
        auto pts = sampler(m, n, a, b, vertices);
		u32 r1 = pcg(pcg(ran));
        for (u32 i = 0; i < pts.n_vertices; i++) 
		{
			f32& x = pts.vertices[2 * i];
            f32& y = pts.vertices[2 * i+1];

            //u32 u = x / a * (1 << 31);
            //u32 v = y / b * (1 << 31);
            //u = scrambler(u, r1);
            //v = scrambler(v, r1);
            //x = u * 0x1p-32f * a;
            //y = v * 0x1p-32f * b;
        }
        return pts;
    }
};


points halton_permute(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(halton, permute_scramble)(m, n, a, b, ran, vertices);
}
points hammersley_permute(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(hammersley, permute_scramble)(m, n, a, b, ran, vertices);
}
points sobol_permute(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(sobol, permute_scramble)(m, n, a, b, ran, vertices);
}
points halton_fast_owen(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(halton, fast_owen_scramble)(m, n, a, b, ran, vertices);
}
points hammersley_fast_owen(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(hammersley, fast_owen_scramble)(m, n, a, b, ran, vertices);
}
points sobol_fast_owen(u32 m, u32 n, f32 a, f32 b, u32 ran, f32* vertices)
{
	return scramble_functor(sobol, fast_owen_scramble)(m, n, a, b, ran, vertices);
}
