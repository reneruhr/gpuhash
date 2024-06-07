#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <cmath>

#include <utility>
#include <numbers>
#include <filesystem>
#include <string>
#include <span>
#include <algorithm>
#include <array>
#include <numeric>
#include <thread>
#include <unordered_map>

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

#define arraysize(x)  sizeof(x) / sizeof(x[0]);

#include "quaternions.h"
#include "common.h"
#include "vec_math.h"
#include "random_int.h"

#include "memory_allocator.cpp"
#include "gl_backend.cpp"
#include "samplers.cpp"
#include "warps.cpp"
#include "statistics.cpp"
#include "colors.cpp"
#include "primitives_2d.cpp"
#include "canvas.cpp"
#include "test_functions.cpp"

#include <ft2build.h>
#include FT_FREETYPE_H


#define loc_size 256

const GLuint WIDTH = 3*1024, HEIGHT = 2*1024;
const f32 ratio = (f32)HEIGHT / WIDTH;
vec2_f32 cam_size{ (f32)WIDTH, (f32)HEIGHT};
vec2_f32 scroll_offset = vec2_f32(0);
vec2_f32 awds_offset = vec2_f32(0);
f32 scroll_speed = 300.f;
f32 awds_speed = 100.f;
vec2_u32 mouse_pos;
enum mouse_states
{
	mouse_state_none = 0,
	mouse_state_left_pressed = 1,
	mouse_state_left_keep_pressed = 2,
	mouse_state_left_release = 4
} mouse_state;


arena arena0(1*gigabyte);  
arena arena1(1*gigabyte);  


// events 
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mousebutton_callback(GLFWwindow* window, int button, int action, int mode);
void scroll_callback(GLFWwindow* window, double x, double y);
struct key_event { int key; int action; };
const u32 n_queue{ 1024 };
struct events
{
	u32 begin{};
	u32 end{};
	key_event queue[n_queue];
} events{};
void register_event(key_event event);
bool has_events();
key_event poll_event();
void event_loop(GLFWwindow* window);

struct per_frame_data
{
	quat q{ 1. };
	vec3_f32 translation{0.};
	f32  scale{1};
	vec4_f32 color{1.};
	u32 texture{ 0 }; // and flags
	vec3_f32 pad;
};	

per_frame_data interpolate(per_frame_data a, per_frame_data b, f32 t)
{
	assert(a.texture == b.texture);
	return { .q = slerp(a.q, b.q, t), .translation = lerp(a.translation, b.translation, t), .scale = std::lerp(a.scale, b.scale, t), .color = lerp(a.color, b.color, t), .texture = a.texture};
}

struct scene
{
	u32 n_seconds;
	u32 n_meshes;
	u32 *mesh_table;
	per_frame_data *data;
	opengl_context context;

	f32 rate;
};

mat4_f32 cur_cam()
{
	vec2_f32 cam_offs = scroll_offset+awds_offset;
	mat4_f32 p = orthographic(cam_offs.x,cam_offs.x+cam_size.x, cam_offs.y,cam_offs.y+cam_size.y,-2,2);
	mat4_f32 v = scale(1.f);
	mat4_f32 camera = mul(p, v);
	return camera;
}

#include "gui.cpp"

const u32 max_points   = 1 << 20;
const u32 max_pointset = 2;
opengl_context context_points;
u32 n_pointset{};

opengl_context context_stat_test;
opengl_context context_stat_test_draw;
opengl_context context_sum;


u32 frame_count{ 0 };


struct test_state
{
	enum test_type
	{
		lcg_a = 0,
		shear_u_n
	};
	
	test_type type;
	bool test_running = false;
	bool draw_results = false;
};

test_state lcg_a_test{ test_state::lcg_a,     false, false };
test_state shear_test{ test_state::shear_u_n, false, false };

bool compute_path = true;



const u32 max_time = 128u;

const u32 max_lcg_a= 256u;

const u32 max_shear= 128u;

const u32 calc_per_path = 32u;

struct parallel_sum_result
{
	u32  n;
	u32* sums;
} parallel_sum_result;


struct shear_test_result
{
	u32 n = max_shear * max_shear;
	f32 KL[max_shear * max_shear];
} shear_test_result;

struct per_frame_data_points
{
	vec3_f32 translation{0.};
	f32  scale{1};
	vec4_f32 color{1.};
};	



struct alignas(64) torus_control
{
	u32 bits = 11;
	u32 initial = 5;
	u32 lcg_a = 17;
	u32 lcg_c = 483;

	u32 shear_u = 5;
	u32 shear_n = 5;
	u32 shear_u2 = 5;
	u32 shear_n2 = 5;

	u32 apply_rng = 1;
	u32 xorshifts = 1 | 2;
	u32 time = 0;
	u32 pad = 0;
} torus;	

vec2_u32 torus_pos{ 50,50 };





struct torus_control_hash
{
    std::size_t operator()(const torus_control& t) const noexcept
    {
        std::size_t hash = 0;
        hash_combine(hash, t.bits, t.initial, t.lcg_a, t.lcg_c, t.shear_u, t.shear_n, t.shear_u2, t.shear_n2, t.apply_rng, t.xorshifts, t.time, t.pad);
        return hash;
    }

private:
    template <typename T>
    void hash_combine(std::size_t& seed, const T& value) const
    {
        seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template <typename T, typename... Rest>
    void hash_combine(std::size_t& seed, const T& value, const Rest&... rest) const
    {
        hash_combine(seed, value);
        (hash_combine(seed, rest), ...);
    }
};

struct torus_control_equal
{
    bool operator()(const torus_control& lhs, const torus_control& rhs) const
    {
        return lhs.bits == rhs.bits && lhs.initial == rhs.initial && lhs.lcg_a == rhs.lcg_a && lhs.lcg_c == rhs.lcg_c &&
               lhs.shear_u == rhs.shear_u && lhs.shear_n == rhs.shear_n && lhs.shear_u2 == rhs.shear_u2 && lhs.shear_n2 == rhs.shear_n2 &&
               lhs.apply_rng == rhs.apply_rng && lhs.xorshifts == rhs.xorshifts && lhs.time == rhs.time && lhs.pad == rhs.pad;
    }
};

std::unordered_map<torus_control, f32, torus_control_hash, torus_control_equal> KLs;


void init_torus_rng()
{

	auto program = compile_shaders(R"(
										#version 460 core 

										struct per_frame_data 
										{
											vec3 translation;
											float scale;
											vec4 color;
										}; 

										struct torus_control
										{
											uint bits;
											uint initial;
											uint lcg_a;
											uint lcg_c;

											uint shear_u;
											uint shear_n;
											uint shear_u2;
											uint shear_n2;

											uint apply_rng;
											uint xorshifts;
											uint time;
											uint pad;
										};	

										layout(std430, binding = 0) restrict readonly buffer vertices
										{
											uint in_vertices[];
										};

										layout(std430, binding = 1) restrict readonly buffer colors
										{
											vec4 in_colors[];
										};

										layout(std430, binding = 2) restrict readonly buffer meshids
										{
											unsigned int in_meshid[];
										};

										layout(std430, binding = 3) restrict readonly buffer per_frame_datas
										{
											per_frame_data pfd[];
										};

										layout (std140, binding = 0) uniform camera 
										{
											mat4 cam;
										};

										layout (std140, binding = 1) uniform torus_uniform
										{
											torus_control torus;
										};

										layout (location=0) out vec4 frag_color;

										// pcg2d-like
										// https://www.shadertoy.com/view/XlGcRh
										uvec2 apply_rng(uvec2 v)
										{
											v = v * torus.lcg_a + torus.lcg_c;

											v.x += v.y * torus.shear_u;
											v.y += v.x * torus.shear_n;
											
											if((torus.xorshifts & 1) == 1)
												v = v ^ (v>>(torus.bits/2));

											v.x += v.y * torus.shear_u2;
											v.y += v.x * torus.shear_n2;

											if((torus.xorshifts & 2) == 2)
												v = v ^ (v>>(torus.bits/2));

											return v & ( (1<<torus.bits) - 1);
										}

										void main(void)
										{
											gl_PointSize = torus.bits < 8 ? 1 : torus.bits-6 ;
											uint id = in_meshid[gl_VertexID];
											uvec2 v = uvec2(in_vertices[gl_VertexID] % (1 << torus.initial), in_vertices[gl_VertexID] / (1 << torus.initial));
											v += uvec2(torus.time * (1 << torus.initial));
											uvec2 p = torus.apply_rng == 1 ? apply_rng(v) : v;
											vec3 pos = vec3(pfd[id].scale * vec2(p), 0) + pfd[id].translation;
											gl_Position = cam*vec4(pos, 1.0);
											frag_color = pfd[id].color * in_colors[gl_VertexID];	
										}
										)",
										R"(
										#version 460 core 
										layout (location=0) in  vec4 frag_color;
										layout (location=0) out vec4 color;
										void main(void)
										{
											color = frag_color;
											float gamma_inverse = 1./2.2;
											color.rgb = pow(color.rgb, vec3(gamma_inverse));
										}
										)");

	storage_info storages_temp_points[] =
	{
	{.binding = 0, .size = max_points * 1 * sizeof(u32), .data = 0},
	{.binding = 1, .size = max_points * 4 * sizeof(f32), .data = 0},
	{.binding = 2, .size = max_points * 1 * sizeof(u32), .data = 0},
	{.binding = 3, .size = max_pointset * 1 * sizeof(per_frame_data_points) ,.data = 0},
	};

	u32 n_storages_points = sizeof(storages_temp_points) / sizeof(storage_info);
	storage_info* storages_points = alloc_n<storage_info>(arena0, n_storages_points);
	memcpy(storages_points, storages_temp_points, sizeof(storages_temp_points));

	for (u32 u{}; u < n_storages_points; u++)
	{
		auto& storage = storages_points[u] ;
		storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
		bind_storage_buffer(storage.buffer, storage.binding);
	}

	context_points =
	{
		.program = program,
		.mode = GL_POINTS,
		.first = 0,
		.count = 0,
		.draw_mode = opengl_context::draw_mode::array,
		.vao      = create_vao(),
		.uniform  = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.uniform2 = create_buffer(sizeof(torus_control), GL_DYNAMIC_DRAW),
		.n_storages = n_storages_points,
		.storages = storages_points
	};
}

void add_points(points_u32 pts, vec3_f32 pos, f32 scale)
{
	assert(context_points.count+pts.n_vertices <= max_points);

	u32* ids = alloc_n<u32>(arena0, pts.n_vertices);
	std::fill(ids, ids + pts.n_vertices, n_pointset);
	per_frame_data_points pfd{};
	pfd.translation = pos;
	pfd.scale       = scale;

	vec4_f32 *colors_ = alloc_n<vec4_f32>(arena0, pts.n_vertices);
	std::fill(colors_, colors_ + pts.n_vertices, colors[black]);

	update_buffer_storage(context_points.storages[0].buffer, pts.n_vertices*sizeof(u32)*1,		pts.vertices, context_points.count*1*sizeof(u32));
	update_buffer_storage(context_points.storages[1].buffer, pts.n_vertices*sizeof(f32)*4,		colors_,      context_points.count*4*sizeof(f32));
	update_buffer_storage(context_points.storages[2].buffer, pts.n_vertices*sizeof(u32),		ids,          context_points.count*1*sizeof(u32));
	update_buffer_storage(context_points.storages[3].buffer, sizeof(per_frame_data_points),     &pfd,		  n_pointset*sizeof(per_frame_data_points));
	context_points.count += pts.n_vertices;
}

void draw_torus_rng()
{
	for (u32 u{}; u < context_points.n_storages; u++) 
		bind_storage_buffer(context_points.storages[u].buffer, context_points.storages[u].binding);

	auto cam = cur_cam();
	map_buffer((void*)cam.m,   sizeof(mat4_f32), context_points.uniform);
	bind_uniform_block(context_points.uniform, 0);

	map_buffer((void*)&torus,   sizeof(torus_control), context_points.uniform2);
	bind_uniform_block(context_points.uniform2, 1);
	context_points.count = ( 1<< (2*torus.initial));

	draw(&context_points);
}

void compute_torus_rng()
{
	for (u32 u{}; u < context_stat_test_draw.n_storages; u++) 
		bind_storage_buffer(context_stat_test_draw.storages[u].buffer, context_stat_test_draw.storages[u].binding);

	auto cam = cur_cam();
	map_buffer((void*)cam.m,   sizeof(mat4_f32), context_stat_test_draw.uniform);
	bind_uniform_block(context_stat_test_draw.uniform, 0);

	map_buffer((void*)&torus,   sizeof(torus_control), context_stat_test_draw.uniform2);
	bind_uniform_block(context_stat_test_draw.uniform2, 1);

	context_stat_test.n_work_groups_x = (1u<<(2*torus.initial))/ loc_size,
	glUseProgram(context_stat_test.program);
	glDispatchCompute(context_stat_test.n_work_groups_x, context_stat_test.n_work_groups_y, context_stat_test.n_work_groups_z);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);	

	context_stat_test_draw.count = (1 << (2 * torus.initial));
	draw(&context_stat_test_draw);
}

void analyze_distribution()
{
	f64 rev_sq = 1./std::sqrt(static_cast<f64>(parallel_sum_result.n));

	f64* Sn = alloc_n<f64>(arena1, max_time);
	for (u32 u{}; u < max_time; u++)
			Sn[u] = (2. * parallel_sum_result.sums[u] - static_cast<f64>(parallel_sum_result.n)) * rev_sq ;
	
	std::sort(Sn, Sn + max_time);

	u32 n_bars{ 30 };

	f32 min_bar = -3.;
	f32 max_bar =  3.;

	f32 bar_width = (max_bar - min_bar) / (n_bars-1);

	f32* histogram_empirical = alloc_n<f32>(arena1, n_bars+1);
	f32* histogram_normal    = alloc_n<f32>(arena1, n_bars+1);

	auto norm_cdf = [](f64 x) { return 0.5 * std::erfc(-x / std::sqrt(2.));  };

	for (u32 u{1}; u< n_bars; u++)
	{
		histogram_normal[u] = norm_cdf(min_bar + u * bar_width) - norm_cdf(min_bar + (u - 1) * bar_width);
	}

	histogram_normal[0]          =     norm_cdf(min_bar);
	histogram_normal[n_bars]     = 1 - norm_cdf(max_bar);

	for (u32 u{}; u < n_bars+1; u++) 
		histogram_empirical[u] = 0;

	for (u32 u{}, j{}; u < max_time;)
	{
		if (Sn[u] < min_bar + j * bar_width)
		{
			histogram_empirical[j]++;
			u++;
		}
		else if(j < n_bars-1)
		{
			j++;
		}
		else	
		{
			histogram_empirical[n_bars] = (max_time - u);
			break;
		}
	}

	for (u32 u{}; u <= n_bars; u++)
		histogram_empirical[u] /= max_time;

/*
	for (u32 u{}; u < n_bars+1; u++)
	{
		for (u32 k{}; k < 1000 * histogram_empirical[u]; k++) printf("#");
		printf("\n");
	}
	printf("\n Normal:\n");
	for (u32 u{}; u < n_bars+1; u++)
	{
		for (u32 k{}; k < 1000 * histogram_normal[u]; k++) printf("#");
		printf("\n");
	}
	*/

	f32 KL{};

	for (u32 u{}; u<=n_bars; u++)
	{
		if((histogram_normal[u] > 0.) && (histogram_empirical[u] > 0.))
			KL += histogram_empirical[u] * std::log(histogram_empirical[u] / histogram_normal[u]);
	}
	
	shear_test_result.KL[torus.shear_u + max_shear * torus.shear_n] = KL;
	printf("Kullback Leibler Divergence: %f\n", KL);
}

void compute_parallel_sum()
{
	for (u32 t{}; t < max_time; t++)
	{
		torus.time = t;

		//Compute rng	
		{
			for (u32 u{}; u < context_stat_test_draw.n_storages; u++)
				bind_storage_buffer(context_stat_test_draw.storages[u].buffer, context_stat_test_draw.storages[u].binding);

			auto cam = cur_cam();
			map_buffer((void*)cam.m, sizeof(mat4_f32), context_stat_test_draw.uniform);
			bind_uniform_block(context_stat_test_draw.uniform, 0);

			map_buffer((void*)&torus, sizeof(torus_control), context_stat_test_draw.uniform2);
			bind_uniform_block(context_stat_test_draw.uniform2, 1);

			context_stat_test.n_work_groups_x = (1u << (2 * torus.initial)) / loc_size,
				glUseProgram(context_stat_test.program);
			glDispatchCompute(context_stat_test.n_work_groups_x, context_stat_test.n_work_groups_y, context_stat_test.n_work_groups_z);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}
		
		//Compute sum
		{
			glUseProgram(context_sum.program);

			for (u32 u{}; u < context_sum.n_storages; u++)
				bind_storage_buffer(context_sum.storages[u].buffer, context_sum.storages[u].binding);

			bind_uniform_block(context_sum.uniform, 0);
			bind_uniform_block(context_sum.uniform2, 1);

			context_sum.n_work_groups_x = (1u << (2 * torus.initial)) / loc_size / 2;
			alignas(64) u32 sum_redux_uniform[4] = { (1u << (2 * torus.initial)), (1u << (2 * torus.initial)) / 2, 0u, 0u };

			while (sum_redux_uniform[1] >= 256)
			{
				map_buffer((void*)sum_redux_uniform, 4 * sizeof(u32), context_sum.uniform);
				map_buffer((void*)&torus, sizeof(torus_control), context_sum.uniform2);
				//printf("t=%u, total = %u, Launch compute: Offset %u, Workgroups: %u\n", torus.time, sum_redux_uniform[0], sum_redux_uniform[1], context_sum.n_work_groups_x);
				glDispatchCompute(context_sum.n_work_groups_x, context_sum.n_work_groups_y, context_sum.n_work_groups_z);
				glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

				sum_redux_uniform[1] /= 2;
				context_sum.n_work_groups_x /= 2;
			}
		}

	}

	u32* temp = alloc_n<u32>(arena1, max_time);
	gpu_to_cpu(context_sum.storages[context_sum.n_storages - 1].buffer, temp, max_time * sizeof(u32));
	memcpy(parallel_sum_result.sums, temp, sizeof(u32) * max_time);
	analyze_distribution();
}

void init_stat_test()
{
	auto program = compile_shaders(R"(
										#version 460 core 

										struct torus_control
										{
											uint bits;
											uint initial;
											uint lcg_a;
											uint lcg_c;

											uint shear_u;
											uint shear_n;
											uint shear_u2;
											uint shear_n2;

											uint apply_rng;
											uint xorshifts;
											uint time;
											uint pad;
										};	

										layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

										layout(std430, binding = 0) restrict readonly buffer vertices
										{
											uint in_vertices[];
										};

										layout(std430, binding = 2) restrict readonly buffer meshids
										{
											unsigned int in_meshid[];
										};

										layout(std430, binding = 4) restrict writeonly buffer res
										{
											uvec2 result[];
										};

										layout (std140, binding = 1) uniform torus_uniform
										{
											torus_control torus;
										};

										// pcg2d-like
										// https://www.shadertoy.com/view/XlGcRh
										uvec2 apply_rng(uvec2 v)
										{
											v = v * torus.lcg_a + torus.lcg_c;

											v.x += v.y * torus.shear_u;
											v.y += v.x * torus.shear_n;
											
											if((torus.xorshifts & 1) == 1)
												v = v ^ (v>>(torus.bits/2));

											v.x += v.y * torus.shear_u2;
											v.y += v.x * torus.shear_n2;

											if((torus.xorshifts & 2) == 2)
												v = v ^ (v>>(torus.bits/2));

											return v & ( (1<<torus.bits) - 1);
										}

										void main()
										{
											uint index = gl_GlobalInvocationID.x;

											uint id = in_meshid[index];
											uvec2 v = uvec2(in_vertices[index] % (1 << torus.initial), in_vertices[index] / (1 << torus.initial));
											v += uvec2(torus.time * (1 << torus.initial));
											result[index] =  apply_rng(v);
										}
										)");

	storage_info storages_temp_points[] =
	{
	{.binding = 0, .size = max_points * 1 * sizeof(u32), .data = 0},
	//{.binding = 1, .size = max_points * 4 * sizeof(f32), .data = 0},
	{.binding = 2, .size = max_points * 1 * sizeof(u32), .data = 0},
	//{.binding = 3, .size = max_pointset * 1 * sizeof(per_frame_data_points) ,.data = 0},
	{.binding = 4, .size = max_points * 1 * sizeof(vec2_u32), .data = 0},
	};

	u32 n_storages_points = sizeof(storages_temp_points) / sizeof(storage_info);
	storage_info* storages_points = alloc_n<storage_info>(arena0, n_storages_points);
	memcpy(storages_points, storages_temp_points, sizeof(storages_temp_points));

	for (u32 u{}; u < n_storages_points; u++)
	{
		auto& storage = storages_points[u] ;
		if (storage.binding < 4)
			storage.buffer = context_points.storages[storage.binding].buffer;
		else
		{
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}
	}

	context_stat_test =
	{
		.program = program,
		.uniform2 = create_buffer(sizeof(torus_control), GL_DYNAMIC_DRAW),
		.n_storages = n_storages_points,
		.storages = storages_points,
		.n_work_groups_x = (1u<<(2*torus.initial))/ loc_size,
		.n_work_groups_y = 1,
		.n_work_groups_z = 1,
		.local_size_x = loc_size,
		.local_size_y = 1,
		.local_size_z = 1,
	};

}

void init_stat_test_draw()
{

	auto program = compile_shaders(R"(
										#version 460 core 

										struct per_frame_data 
										{
											vec3 translation;
											float scale;
											vec4 color;
										}; 

										struct torus_control
										{
											uint bits;
											uint initial;
											uint lcg_a;
											uint lcg_c;

											uint shear_u;
											uint shear_n;
											uint shear_u2;
											uint shear_n2;

											uint apply_rng;
											uint xorshifts;
											uint time;
											uint pad;
										};	

										layout(std430, binding = 0) restrict readonly buffer vertices
										{
											uint in_vertices[];
										};

										layout(std430, binding = 1) restrict readonly buffer colors
										{
											vec4 in_colors[];
										};

										layout(std430, binding = 2) restrict readonly buffer meshids
										{
											unsigned int in_meshid[];
										};

										layout(std430, binding = 3) restrict readonly buffer per_frame_datas
										{
											per_frame_data pfd[];
										};

										layout(std430, binding = 4) restrict readonly buffer results 
										{
											uvec2 result[];
										};

										layout (std140, binding = 0) uniform camera 
										{
											mat4 cam;
										};

										layout (std140, binding = 1) uniform torus_uniform
										{
											torus_control torus;
										};

										layout (location=0) out vec4 frag_color;

										void main(void)
										{
											gl_PointSize = torus.bits < 8 ? 1 : torus.bits-6 ;
											uint id = in_meshid[gl_VertexID];
											uvec2 v = uvec2(in_vertices[gl_VertexID] % (1 << torus.initial), in_vertices[gl_VertexID] / (1 << torus.initial));
											v += uvec2(torus.time * (1 << torus.initial));
											uvec2 p = torus.apply_rng == 1 ? result[gl_VertexID] : v;
											vec3 pos = vec3(pfd[id].scale * vec2(p), 0) + pfd[id].translation;
											gl_Position = cam*vec4(pos, 1.0);
											frag_color = pfd[id].color * in_colors[gl_VertexID];	
										}
										)",
										R"(
										#version 460 core 
										layout (location=0) in  vec4 frag_color;
										layout (location=0) out vec4 color;
										void main(void)
										{
											color = frag_color;
											float gamma_inverse = 1./2.2;
											color.rgb = pow(color.rgb, vec3(gamma_inverse));
										}
										)");

	storage_info storages_temp_points[] =
	{
	{.binding = 0, .size = max_points * 1 * sizeof(u32), .data = 0},
	{.binding = 1, .size = max_points * 4 * sizeof(f32), .data = 0},
	{.binding = 2, .size = max_points * 1 * sizeof(u32), .data = 0},
	{.binding = 3, .size = max_pointset * 1 * sizeof(per_frame_data_points) ,.data = 0},
	{.binding = 4, .size = max_points * 1 * sizeof(vec2_u32), .data = 0},
	};

	u32 n_storages_points = sizeof(storages_temp_points) / sizeof(storage_info);
	storage_info* storages_points = alloc_n<storage_info>(arena0, n_storages_points);
	memcpy(storages_points, storages_temp_points, sizeof(storages_temp_points));

	for (u32 u{}; u < n_storages_points; u++)
	{
		auto& storage = storages_points[u];
		if (storage.binding < 4)
			storage.buffer = context_points.storages[storage.binding].buffer;
		else
			storage.buffer = context_stat_test.storages[context_stat_test.n_storages-1].buffer;
	}

	context_stat_test_draw =
	{
		.program = program,
		.mode = GL_POINTS,
		.first = 0,
		.count = 0,
		.draw_mode = opengl_context::draw_mode::array,
		.vao      = create_vao(),
		.uniform  = create_buffer(sizeof(mat4_f32), GL_DYNAMIC_DRAW),
		.uniform2 = create_buffer(sizeof(torus_control), GL_DYNAMIC_DRAW),
		.n_storages = n_storages_points,
		.storages = storages_points
	};
}



void init_parallel_sum()
{
	auto program = compile_shaders(R"(
										#version 460 core 

										layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

										struct torus_control
										{
											uint bits;
											uint initial;
											uint lcg_a;
											uint lcg_c;

											uint shear_u;
											uint shear_n;
											uint shear_u2;
											uint shear_n2;

											uint apply_rng;
											uint xorshifts;
											uint time;
											uint pad;
										};	

										layout(std430, binding = 4) restrict readonly buffer input_rng
										{
											uvec2 in_rng[];
										};

										layout(std430, binding = 5) restrict buffer out_sum_buffer
										{
											uint out_sum[];
										};

										layout(std430, binding = 6) restrict writeonly buffer final_sum_buffer 
										{
											uint final_sum[];
										};


										layout (std140, binding = 0) uniform sum_redux 
										{
											uint total;
											uint offset;
											uint pad0;
											uint pad1;
										};

										layout (std140, binding = 1) uniform torus_uniform
										{
											torus_control torus;
										};


										shared uint sum[256];

										uint sum_bits(uint a)
										{
											uint b = 0;
											for(uint i = 0;i<torus.bits; i++) b += (a>>i) & 1;
											return b;
										}


										void main()
										{
											uint index = gl_GlobalInvocationID.x;
											uint local_index = gl_LocalInvocationID.x;
											
											if(offset == total / 2)
											{
												out_sum[index] = sum_bits(in_rng[index].x) + sum_bits(in_rng[index].y) + sum_bits(in_rng[index+offset].x) + sum_bits(in_rng[index+offset].y);
											}
											else if(offset > 256)
											{
												out_sum[index] += out_sum[index+offset];
											}
											else
											{
												sum[local_index] = out_sum[local_index] + out_sum[local_index+256];
												barrier();	
												if(local_index < 128)sum[local_index] += sum[local_index+128];
												barrier();
												if(local_index < 64) sum[local_index] += sum[local_index+64];
												barrier();
												if(local_index < 32) sum[local_index] += sum[local_index+32];
												barrier();
												if(local_index < 16) sum[local_index] += sum[local_index+16];
												barrier();
												if(local_index <  8) sum[local_index] += sum[local_index+ 8];
												barrier();
												if(local_index <  4) sum[local_index] += sum[local_index+ 4];
												barrier();
												if(local_index <  2) sum[local_index] += sum[local_index+ 2];
												barrier();	
												
												if(local_index == 0) final_sum[torus.time] = sum[0] + sum[1];
											}
										}
										)");

	storage_info storages_temp_points[] =
	{
	//{.binding = 0, .size = max_points * 1 * sizeof(u32), .data = 0},
	//{.binding = 1, .size = max_points * 4 * sizeof(f32), .data = 0},
	//{.binding = 2, .size = max_points * 1 * sizeof(u32), .data = 0},
	//{.binding = 3, .size = max_pointset * 1 * sizeof(per_frame_data_points) ,.data = 0},
	{.binding = 4, .size = max_points * 1 * sizeof(vec2_u32), .data = 0},
	{.binding = 5, .size = max_points * 1 * sizeof(u32), .data = 0},
	{.binding = 6, .size = max_time   * 1 * sizeof(u32), .data = 0},
	};

	u32 n_storages_points = sizeof(storages_temp_points) / sizeof(storage_info);
	storage_info* storages_points = alloc_n<storage_info>(arena0, n_storages_points);
	memcpy(storages_points, storages_temp_points, sizeof(storages_temp_points));

	for (u32 u{}; u < n_storages_points; u++)
	{
		auto& storage = storages_points[u];

		if (storage.binding == 4)
		{
			storage.buffer = context_stat_test.storages[context_stat_test.n_storages-1].buffer;
		}
		else if(storage.binding == 5)
		{
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
		}
		else if(storage.binding == 6)
		{
			storage.buffer = buffer_storage(storage.size, storage.data, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);
			bind_storage_buffer(storage.buffer, storage.binding);
			u32* temp = alloc_n<u32>(arena0, max_time);
			memset(temp, 0, sizeof(u32) * max_time);
			update_buffer_storage(storage.buffer, sizeof(u32)*max_time, temp, 0);
		}
	}

	context_sum =
	{
		.program = program,
		.uniform =  create_buffer(4*sizeof(u32), GL_DYNAMIC_DRAW),
		.uniform2 = create_buffer(sizeof(torus_control), GL_DYNAMIC_DRAW),
		.n_storages = n_storages_points,
		.storages = storages_points,
		.n_work_groups_x = (1u<< (2*torus.initial))/ loc_size,
		.n_work_groups_y = 1,
		.n_work_groups_z = 1,
		.local_size_x = loc_size,
		.local_size_y = 1,
		.local_size_z = 1,
	};



	parallel_sum_result.n = 2u * torus.bits * (1u << (2 * torus.initial));
	parallel_sum_result.sums = alloc_n<u32>(arena0, max_time);
	for (u32 u{}; u < max_time; u++) parallel_sum_result.sums[u] = u;

}



void draw_graph_lcg_a(graph_canvas& canvas)
{
	static bool init = false;
	
	if(!init)
	{
		u32 n = max_lcg_a;
		f32* x = alloc_n<f32>(arena0, n);
		f32* y = alloc_n<f32>(arena0, n);

		for(const auto& t : KLs)
		{
			x[t.first.lcg_a-1] = (f32)t.first.lcg_a;
			y[t.first.lcg_a-1] = t.second;
			printf("(%f,%f)\n", x[t.first.lcg_a - 1], y[t.first.lcg_a - 1]);
		}

		canvas.add_background(0, 0, arena0);
		canvas.add_graph(x, y, n, graph_canvas::regular, vec2_f32((f32)max_lcg_a, 2.f), vec2_f32(0), vec2_f32(0), black, arena0);
		canvas.add_axis(0, 0, arena0);
		
		f32 xtags[] = { 8./max_lcg_a,32./max_lcg_a, 64./max_lcg_a, 96./max_lcg_a, 128./max_lcg_a, 192./max_lcg_a, 256./max_lcg_a };
		f32 ytags[] = { 0.1/2, 0.2/2, 0.3/2, 0.4/2, 0.5/2, 0.6/2};
		canvas.add_tags(xtags, ytags, 7, 6, { 0,0 }, arena0);

		init = true;
	}

}

void draw_graph_shear(graph_canvas& canvas)
{
	static bool init = false;
	
	if(!init)
	{
		const u32 n = max_shear * max_shear;
		u8* data = alloc_n<u8>(arena0, n*4);
		const u32 pixel_size{ 16 };
		auto mesh = rectangle(pixel_size*max_shear, pixel_size*max_shear, arena0);

		f32 range = *std::max_element(shear_test_result.KL, shear_test_result.KL+shear_test_result.n);
		printf("Max element %f\n", range);

		for (u32 i{}; i < max_shear; i++)
		for (u32 j{}; j < max_shear; j++)
		{
			u8 fval = std::min( static_cast<u32>(255u * shear_test_result.KL[i+j*max_shear]/range), 255u);
			data[4 * (i + j * max_shear)]     = fval;
			data[4 * (i + j * max_shear) + 1] = fval;
			data[4 * (i + j * max_shear) + 2] = fval;
			data[4*  (i + j * max_shear) + 3] = 255;
		}

		mesh.texture = { data, n , max_shear, max_shear};
		canvas.add_shape(mesh, vec3_f32(500, -2000, 0.0) ,1., arena0);
		
		init = true;
	}


}

void construct_gui()
{
	//auto timer = chrono_timer_scoped("\tconstruct gui");
	static auto timer = chrono_timer("fps counter");
	static bool init = false;
	if(!init)
	{
		init = true;
		timer.start();
	}

	clear_gui();

	add_box({ 1u << torus.bits, 1u << torus.bits }, { torus_pos.x, torus_pos.y}, colors[white]);

	u32 grid_box_size = 30;
	u32 grid_width = 200;
	vec2_u32 grid_base = { WIDTH - 800, HEIGHT - 1000 };
	vec2_u32 gridsize = { 1,30 };

	u32 padding = 1;
	auto grid_coords = square_grid(grid_box_size, gridsize.x, gridsize.y, padding, 0);

	add_box({ gridsize.x * (grid_width + 2*padding), gridsize.y * (grid_box_size + 2*padding) }, grid_base+ vec2_u32(25,0), colors[SandyBrown]);

	u32 u{};
	if (button({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[LightYellow], colors[black], "Apply rng"))
		torus.apply_rng = torus.apply_rng == 0 ? 1 : 0;
	pop_tree();

	if (button({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[LightYellow], colors[black], compute_path ? "Compute path" : "Regular Path"))
		compute_path = !compute_path;
	pop_tree();

	if (button({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[LightYellow], colors[black], lcg_a_test.test_running ? "Stop lcg_a test" : "Start lcg_a test"))
		lcg_a_test.test_running = !lcg_a_test.test_running;
	pop_tree();

	if (button({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[LightYellow], colors[black], shear_test.test_running ? "Stop shear test" : "Start shear test"))
	{
		torus.shear_n = 0;
		torus.shear_u = 0;
		shear_test.test_running = !shear_test.test_running;
	}
	pop_tree();



	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "Bits");
	torus.bits    = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.bits,     4, 12,  gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "Initial");
	torus.initial = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.initial,  1, 10,  gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "lcg_a");
	torus.lcg_a   = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.lcg_a,    1, max_lcg_a, gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "lcg_c");
	torus.lcg_c   = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.lcg_c,    0, 1000,gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "shear_u");
	torus.shear_u = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.shear_u,  0, max_shear-1, gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "shear_n");
	torus.shear_n = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.shear_n,  0, max_shear-1, gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "shear_u2");
	torus.shear_u2= uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.shear_u2, 0, max_shear-1, gui_flag_sli_add);
	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "shear_n2");
	torus.shear_n2= uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.shear_n2, 0, max_shear-1, gui_flag_sli_add);
	
	bool xorshift1 = toggle({ grid_box_size, grid_box_size }, grid_coords[u] + grid_base, torus.xorshifts & 1);
	if ((torus.xorshifts & 1) != xorshift1) torus.xorshifts ^= 1;
	add_textbox({ grid_width-grid_box_size-5, grid_box_size }, grid_coords[u++] + grid_base+vec2_u32(grid_box_size+5,0), colors[SandyBrown], colors[black], std::to_string(torus.xorshifts) + "xorshift1");

	bool xorshift2 = toggle({ grid_box_size, grid_box_size }, grid_coords[u] + grid_base, torus.xorshifts & 2);
	if ((torus.xorshifts & 2) != xorshift2 << 1) torus.xorshifts ^= 2;
	add_textbox({ grid_width-grid_box_size-5, grid_box_size }, grid_coords[u++] + grid_base+vec2_u32(grid_box_size+5,0), colors[SandyBrown], colors[black], std::to_string(torus.xorshifts) + "xorshift2");


	add_textbox({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[SandyBrown], colors[black], "time");
	torus.time = uint_slider({ grid_width, grid_box_size }, grid_coords[u++] + grid_base, colors[Orange], torus.time, 0, max_time, gui_flag_sli_add);

	auto intersect_id = intersect_gui(mouse_pos);
	update_hover_gui(intersect_id);

	std::string mouse_button;
	switch (mouse_state)
	{
		case mouse_state_left_pressed:
			mouse_button = "press"; break;
		case mouse_state_left_keep_pressed:
			mouse_button = "pressed"; break;
		case mouse_state_left_release:
			mouse_button = "release"; break;
		default:
			mouse_button = "none";
	}
	std::string mouse = "( " + std::to_string(mouse_pos.x) + ", " + std::to_string(mouse_pos.y) + ") " + mouse_button;
	std::string id = intersect_id != ~0 ? std::to_string(intersect_id) : "none";
	std::string fps = std::to_string(frame_count * 1e9 / timer.stop());
	add_textbox( { 400, 100 }, { WIDTH - 500, HEIGHT - 100 }, vec4_f32(0), colors[Orange], mouse);
	add_textbox( { 400, 100 }, { WIDTH - 500, HEIGHT - 300 }, vec4_f32(0), colors[white], id);
	add_textbox( { 400, 100 }, { WIDTH - 500, HEIGHT - 500 }, vec4_f32(0), colors[white], fps);
	check_gui();
	upload_gui_data();
}

int main(void)
{
	auto window = create_window(WIDTH, HEIGHT, "GUI");

	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetMouseButtonCallback(window, mousebutton_callback);

	glViewport(0, 0, WIDTH, HEIGHT);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glLineWidth(3.);
	const float far_value = 1.0f;
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  


	u32 n = 1 << 20;
	points_u32 pts;
	pts.n_vertices = n;
	pts.vertices = alloc_n<u32>(arena0, n);

	for (u32 u{}; u<n; u++)
	{
		pts.vertices[u]   = u;
	}

	init_gui_context();
	init_torus_rng();
	init_stat_test();
	init_stat_test_draw();
	add_points(pts, {(f32)torus_pos.x, (f32)torus_pos.y, 0.5}, 1.);

	init_parallel_sum();

    graph_canvas canvas({ 500.f,800.f }, { 2300.f, -500.f }, arena0);

    while (!glfwWindowShouldClose(window)) 
	{
		frame_count++;
		//auto timer = chrono_timer_scoped("main loop");
		event_loop(window);

        glClearBufferfv(GL_COLOR, 0, &colors[grey].x);
		glClearBufferfv(GL_DEPTH, 0, &far_value);

		
		construct_gui();

		draw_gui();
	
		if (compute_path)
		{
			for (u32 k{}; k < calc_per_path; k++)
			{
				if (lcg_a_test.test_running)
				{
					compute_parallel_sum();
					if (torus.lcg_a >= max_lcg_a)
					{
						torus.lcg_a = 1;
						lcg_a_test.test_running = false;
						lcg_a_test.draw_results = true;
						draw_graph_lcg_a(canvas);
					}
					else
						torus.lcg_a++;
				}
				else if (shear_test.test_running)
				{
					compute_parallel_sum();
					torus.shear_n++;
					if (torus.shear_n >= max_shear)
					{
						torus.shear_n = 0;
						torus.shear_u++;
					}

					if (torus.shear_u >= max_shear)
					{
						torus.shear_n = 0;
						torus.shear_u = 0;
						shear_test.test_running = false;
						shear_test.draw_results = true;
						draw_graph_shear(canvas);
					}
				}
			}
			compute_torus_rng();
		}
		else
			draw_torus_rng();

		if(lcg_a_test.draw_results || shear_test.draw_results)
			canvas.draw_canvas(cur_cam());

		glfwSwapBuffers(window);

		arena1.clear();
	}

    glfwTerminate();


    return 0;
}

















void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	else if (action == GLFW_RELEASE)
		register_event({ key, action });
}


void mousebutton_callback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
		mouse_state = mouse_state & (mouse_state_left_pressed | mouse_state_left_keep_pressed) ? mouse_state_left_keep_pressed : mouse_state_left_pressed;
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		mouse_state = mouse_state_left_release;
	else
		mouse_state = mouse_state_none;
}

void scroll_callback(GLFWwindow* window, double x, double y)
{
	scroll_offset.x += scroll_speed*x;
	scroll_offset.y += scroll_speed*y;
}


void register_event(key_event event)
{
	events.queue[events.end] = event;
	events.end = (events.end + 1u) % n_queue;
}

bool has_events()
{
	return events.begin != events.end;
}

void event_loop(GLFWwindow* window)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	
	mouse_pos = vec2_u32(xpos, HEIGHT - ypos) + vec2_u32(awds_offset.x, awds_offset.y) + vec2_u32(scroll_offset.x, scroll_offset.y);

	if (mouse_state & mouse_state_left_pressed)
		mouse_state = mouse_state_left_keep_pressed;
	else if (mouse_state & mouse_state_left_release)
		mouse_state = mouse_state_none;

	glfwPollEvents();

	while (has_events())
	{
		auto event = poll_event();
		if (event.key == GLFW_KEY_R)
		{
		}
		else if (event.key == GLFW_KEY_T)
		{
		}
		else if (event.key == GLFW_KEY_ENTER)
		{
		}
		else if (event.key == GLFW_KEY_UP)
		{
		}
		else if (event.key == GLFW_KEY_DOWN)
		{
		}
		else if (event.key == GLFW_KEY_RIGHT)
		{
		}
		else if (event.key == GLFW_KEY_LEFT)
		{
		}
		else if (event.key == GLFW_KEY_SPACE)
		{
		}
		else if (event.key == GLFW_KEY_A)
		{
			awds_offset.x -= awds_speed;
		}
		else if (event.key == GLFW_KEY_W)
		{
			awds_offset.y += awds_speed;
		}
		else if (event.key == GLFW_KEY_S)
		{
			awds_offset.y -= awds_speed;
		}
		else if (event.key == GLFW_KEY_D)
		{
			awds_offset.x += awds_speed;
		}
		else if (event.key == GLFW_KEY_PAGE_UP)
		{
			awds_offset.y += 100*awds_speed;
		}
		else if (event.key == GLFW_KEY_PAGE_DOWN)
		{
			awds_offset.y -= 100*awds_speed;
		}
	}
}

key_event poll_event()
{
	assert(has_events() && "polled for events but no events");
	key_event event = events.queue[events.begin];
	events.begin = (events.begin + 1u) % n_queue;
	return event;
}
