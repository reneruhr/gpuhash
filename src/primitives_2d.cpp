mesh rectangle(f32 a, f32 b, arena& arena)
{
	u32 n_vertices = 4;
	f32 *vertices  = alloc_n<f32>(arena, 2*n_vertices);
	f32 *colors    = alloc_n<f32>(arena, 4*n_vertices);
	f32 *uv        = alloc_n<f32>(arena, 2*n_vertices);
	f32 vertices_[] =
	{
		0.f,0.f,
		a,0.f,
		a, b,
		0, b,
	};
	u32 n_indicies = 6;
	u32 *indicies = alloc_n<u32>(arena, n_indicies);
	u32 indicies_[] = 
	{
		0,1,3, 
		1,2,3
	};
	memcpy(vertices, vertices_, sizeof(vertices_));
	memcpy(indicies, indicies_, sizeof(indicies_));
	for(u32 u{}; u<n_vertices; u++)
	{
		colors[4 * u] = 1.f;
		colors[4*u+1] = 1.f;
		colors[4*u+2] = 1.f;
		colors[4*u+3] = 1.f;

		uv[2 * u    ] = vertices[2 * u + 0] ? 1: 0;
		uv[2 * u + 1] = vertices[2 * u + 1] ? 1: 0;
	}

	return mesh{vertices, n_vertices, indicies, n_indicies, colors, uv};
}

mesh disk(f32 r, u32 s, arena& arena)
{
	u32 n_vertices = s+1;
	u32 n_indices = 3*s;
	f32 *vertices  = alloc_n<f32>(arena, 2*n_vertices);
	f32 *colors    = alloc_n<f32>(arena, 4*n_vertices);
	f32 *uv        = alloc_n<f32>(arena, 2*n_vertices);
	u32 *indices  = alloc_n<u32>(arena, n_indices);

	vertices[0] = 0.f;
	vertices[1] = 0.f;
	uv[0] = .5f;
	uv[1] = .5f;
	for(u32 u{}; u< s; u++)
	{
		vertices[2+2*u  ] = std::cos(tau*u/s);
		vertices[2+2*u+1] = std::sin(tau*u/s);

		indices[3*u  ] = 0;
		indices[3*u+1] = 1+ u;
		indices[3*u+2] = 1+(u+1)%s;

		uv[2+2 * u    ] = vertices[2 + 2 * u] / 2 + .5f;
		uv[2+2 * u + 1] = vertices[2 + 2 * u+1] / 2 + .5f;

		vertices[2 + 2 * u]     *= r;
		vertices[2 + 2 * u + 1] *= r;
	}

	for(u32 u{}; u<n_vertices; u++)
	{
		colors[4*u]   = 1.f;
		colors[4*u+1] = 1.f;
		colors[4*u+2] = 1.f;
		colors[4*u+3] = 1.f;
	}

	return mesh{vertices, n_vertices, indices, n_indices, colors, uv};
}



mesh circular_segment(f32 r, f32 phi, u32 s, arena& arena)
{
	u32 n_vertices = s + 1;
	u32 n_indicies = 3 * s;
	f32 *vertices  = alloc_n<f32>(arena, 2*n_vertices);
	f32 *colors    = alloc_n<f32>(arena, 4*n_vertices);
	f32 *uv        = alloc_n<f32>(arena, 2*n_vertices);
	u32 *indicies  = alloc_n<u32>(arena, n_indicies);

	vertices[0] = std::cos(phi / 2) * std::cos(phi / 2);
	vertices[1] = std::sin(phi / 2) * std::cos(phi / 2);
	uv[0] = vertices[0] / 2 + .5f;
	uv[1] = vertices[1] / 2 + .5f;

	vertices[2*s]   = std::cos(phi);
	vertices[2*s+1] = std::sin(phi);
	for(u32 u{}; u < s-1; u++)
	{
		vertices[2+2*u  ] = std::cos(phi*u/(s-1));
		vertices[2+2*u+1] = std::sin(phi*u/(s-1));

		indicies[3*u  ] = 0;
		indicies[3*u+1] = 1+u;
		indicies[3*u+2] = 2+u;

		uv[2+2 * u    ] = vertices[2 + 2 * u] / 2 + .5f;
		uv[2+2 * u + 1] = vertices[2 + 2 * u+1] / 2 + .5f;
	}

	for (u32 u{}; u<s+1; u++)
	{
		vertices[2 * u]     *= r;
		vertices[2 * u + 1] *= r;
	}

	for(u32 u{}; u<n_vertices; u++)
	{
		colors[4*u]   = 1.f;
		colors[4*u+1] = 1.f;
		colors[4*u+2] = 1.f;
		colors[4*u+3] = 1.f;
	}

	return mesh{vertices, n_vertices, indicies, n_indicies, colors, uv};
}
