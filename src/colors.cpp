const u32 n_colors = 26;
vec4_f32 colors[n_colors] =
{
		{1.0, 0.0, 0.0, 1.0}, // Red
		{0.0, 0.5, 0.0, 1.0}, // Green
		{1.0, 0.5, 0.0, 1.0}, // Orange
		{0.5, 0.0, 0.5, 1.0}, // Purple
		{0.0, 0.5, 1.0, 1.0}, // Sky Blue
		{1.0, 0.0, 1.0, 1.0}, // Magenta
		{1.0, 1.0, 0.0, 1.0}, // Yellow
		{0.0, 1.0, 1.0, 1.0}, // Cyan
		{0.5, 0.0, 0.0, 1.0}, // Maroon
		{0.0, 0.5, 0.5, 1.0}, // Teal
		{0.5, 0.5, 0.0, 1.0}, // Olive
		{0.85, 0.75, 0.85, 1.0},  // Lavender
		{0.90, 0.90, 0.98, 1.0},  // Alice Blue
		{1.00, 0.89, 0.77, 1.0},  // Peach
		{0.67, 0.85, 0.90, 1.0},  // Light Blue
		{0.98, 0.93, 0.36, 1.0},  // Light Yellow
		{0.82, 0.41, 0.55, 1.0},  // Puce
		{0.28, 0.82, 0.80, 1.0},  // Turquoise
		{0.93, 0.51, 0.93, 1.0},  // Orchid
		{0.60, 0.80, 0.20, 1.0},  // Olive Green
		{0.95, 0.64, 0.37, 1.0},  // Sandy Brown
	    {0.1, 0.1, 0.1, 1.0},     // grey
	    { 246. / 255, 241. / 255, 241. / 255,1.f }, // bg_blog
		{ 0, 0, 0, 1 }, // black
		{ 1, 1, 1, 1 }, // white 
		{ 0, 1, 0, 1 }, // verygreen 
};

enum cid 
{
	Red = 0,
	Green,
	Orange,
	Purple,
	SkyBlue,
	Magenta,
	Yellow,
	Cyan,
	Maroon,
	Teal,
	Olive,
	Lavender,
	AliceBlue,
	Peach,
	LightBlue,
	LightYellow,
	Puce,
	Turquoise,
	Orchid,
	OliveGreen,
	SandyBrown,
	grey,
	bg_blog,
	black,
	white,
	verygreen,
};
