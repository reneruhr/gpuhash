#pragma once
#include "common.h"
#include "vec_math.h"
#include <algorithm>

struct fquat32
{
	f32 r{1},x{0},y{0},z{0};
/*
	bool operator==(const fquat32& other) const {
        return x == other.x && y == other.y && z == other.z && r == other.r;
    }
*/
	bool operator==(const fquat32& other) const 
	{
		f32 eps = 1e-6f;
        return  std::fabs(x - other.x) < eps &&
				std::fabs(y - other.y) < eps &&
				std::fabs(z - other.z) < eps &&
				std::fabs(r - other.r) < eps ;
    }
	bool operator<(const fquat32& other) const 
	{
		f32 eps = 1e-6f;
		if(x < other.x - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && y < other.y - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && std::fabs(y-other.y)<eps && z < other.z - eps)
			return true;
        else if ( std::fabs(x-other.x)<eps && std::fabs(z-other.z)<eps && std::fabs(y-other.y)<eps  
										   && r < other.r - eps)
			return true;
		return false;
    }
};

inline vec4_f32 to_vec4(fquat32 q)
{
	return { q.r, q.x, q.y, q.z };
}

inline fquat32 to_quat(vec4_f32 q)
{
	return { q.x, q.y, q.z, q.w };
}

struct quat_s32
{
	int r{1},x{0},y{0},z{0};
};

using quat = fquat32;
using quatz = quat_s32;


inline quat im(quat a)
{
	return {0, a.x, a.y, a.z};
}

inline quat operator~(quat a)
{
	return {a.r, -a.x, -a.y, -a.z};
}

inline quat operator*(f32 t, quat a)
{
	return {t*a.r, t*a.x, t*a.y, t*a.z};
}

inline quat operator+(quat a, quat b)
{
	return {a.r+b.r, a.x+b.x, a.y+b.y, a.z+b.z};
}

inline quat operator-(quat a, quat b)
{
	return {a.r-b.r, a.x-b.x, a.y-b.y, a.z-b.z};
}

inline quat operator^(quat a, quat b)
{
	return {0, a.y*b.z-a.z*b.y, b.x*a.z-b.z*a.x, a.x*b.y-a.y*b.x};
} // 6 mults 3 adds

inline f32 operator,(quat a, quat b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
} // 3 mults 3 adds

inline quat operator*(quat a, quat b)
{
	quat q{a.r*b.r - (a,b), 0,0,0};
	return q + a.r*im(b)+b.r*im(a)+ (a^b);
} // 12 mults 14 adds 

inline quat to_quatf(quatz q, f32 norm)
{
	return { q.r*norm, q.x*norm, q.y*norm, q.z*norm};
}

inline quatz im(quatz a)
{
	return {0, a.x, a.y, a.z};
}

inline quatz operator~(quatz a)
{
	return {a.r, -a.x, -a.y, -a.z};
}

inline quatz operator*(int t, quatz a)
{
	return {t*a.r, t*a.x, t*a.y, t*a.z};
}

inline quatz operator+(quatz a, quatz b)
{
	return {a.r+b.r, a.x+b.x, a.y+b.y, a.z+b.z};
}

inline quatz operator^(quatz a, quatz b)
{
	return {0, a.y*b.z-a.z*b.y, b.x*a.z-b.z*a.x, a.x*b.y-a.y*b.x};
}

// Careful: Imaginary dot product
inline int operator,(quatz a, quatz b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline quatz operator*(quatz a, quatz b)
{
	quatz q{a.r*b.r - (a,b), 0,0,0};
	return q + a.r*im(b)+b.r*im(a)+ (a^b);
} 

inline quat make_quat(quatz a, u32 norm)
{
	f32 n = fast_inv_sqrt((f32)norm);
	return { a.r * n, a.x * n, a.y * n, a.z * n };
}

inline f32 norm2(quat q)
{
	return q.r*q.r + q.x*q.x + q.y*q.y + q.z*q.z;
}

inline f32 dot(quat v, quat w)
{
	return v.r*w.r + v.x*w.x + v.y*w.y + v.z*w.z;
}

inline quat normalize(quat q)
{
	return (1.f / std::sqrt(norm2(q))) * q;
}

inline quat normalize_fast(quat q)
{
	return fast_inv_sqrt(norm2(q)) * q;
}

inline void print(quat q)
{
	printf("(%f, %f, %f, %f)\n", q.r,q.x,q.y,q.z);
}

inline void print(quatz q)
{
	printf("(%i, %i, %i, %i)\n", q.r,q.x,q.y,q.z);
}

inline quat angle_axis(f32 angle, f32 x, f32 y, f32 z)
{
	angle /= 2;
	f32 s = std::sin(angle);
	f32 c = std::cos(angle);
	return { c, x * s, y * s, z * s };
}


inline mat4_f32 quat_to_mat4(quat q)
{
	f32 x2 = -2 * q.x * q.x;
	f32 y2 = -2 * q.y * q.y;
	f32 z2 = -2 * q.z * q.z;

	f32 xy = 2 * q.x * q.y;
	f32 zy = 2 * q.z * q.y;
	f32 xz = 2 * q.x * q.z;

	f32 rx = 2 * q.r * q.x;
	f32 ry = 2 * q.r * q.y;
	f32 rz = 2 * q.r * q.z;

	return 
	{	
		1 + y2 + z2,   	xy + rz,	xz - ry, 0,
			xy - rz,1 + x2 + z2,	zy + rx, 0,
			xz + ry,	zy - rx,1 + x2 + y2, 0,
				  0,		  0,		0,	 1
	};
}

// GPU implementation
inline quat quat_mul(quat a, quat b)
{
	return      quat{dot(a,~b),0,0,0} + 
				a.r*im(b) + b.r * im(a) + (im(a) ^ im(b)) ;
}


// Blow 2004 - Understanding Slerp Then Not Using It
quat slerp(quat a, quat b, f32 t) 
{

	f32 dotab = std::clamp(dot(a, b), -1.f, 1.f);

    const f32 DOT_THRESHOLD = 0.9995;
    if (dotab > DOT_THRESHOLD) 
	{
        quat result = a + t * (b - a);
        return normalize(result);
    }

    f32 theta_0 = std::acos(dotab);
    f32 theta = theta_0*t;

    quat q = b - dotab*a;
	q = normalize(q);

    return std::cos(theta)*a + std::sin(theta)*q;
}
