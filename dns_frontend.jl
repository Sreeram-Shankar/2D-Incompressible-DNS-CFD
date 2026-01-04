include("multigrid.jl")
include("dns_backend.jl")
using GLMakie
using Makie
using Colors
using Printf
using LinearAlgebra
using GeometryBasics
using Main.MultigridBackend
using Main.DNSBackend
try
    using NativeFileDialog
catch
end
GLMakie.activate!()

#function to compute vorticity from velocity field
function compute_vorticity(u, v, dx)
    n = size(u, 1)
    omega = zeros(n, n)
    for j in 2:n-1, i in 2:n-1
        dv_dx = (v[i+1, j] - v[i-1, j]) / (2 * dx)
        du_dy = (u[i, j+1] - u[i, j-1]) / (2 * dx)
        omega[i, j] = dv_dx - du_dy
    end
    return omega
end

#function to compute statistics from simulation data
function compute_statistics(u_history, v_history, p_history, div_history, residual_history, dx, dt; rho=1.0, L=1.0)
    n = size(u_history[1], 1)
    steps = length(u_history)
    time = [i * dt for i in 0:steps-1]
    
    #computes statiscs of the fluid flow
    vel_mag = [sqrt.(u_history[i].^2 .+ v_history[i].^2) for i in 1:steps]
    vort = [compute_vorticity(u_history[i], v_history[i], dx) for i in 1:steps]
    vort_mag = [abs.(vort[i]) for i in 1:steps]
    kinetic_energy = [0.5 * sum(u_history[i].^2 .+ v_history[i].^2) * dx^2 for i in 1:steps]
    enstrophy = [0.5 * sum(vort[i].^2) * dx^2 for i in 1:steps]
    total_circulation = [sum(vort[i]) * dx^2 for i in 1:steps]
    mass_flow_in  = [rho * dx * sum(u_history[i][:, 1]) for i in 1:steps]
    mass_flow_out = [rho * dx * sum(u_history[i][:, end]) for i in 1:steps]

    return (time=time, vel_mag=vel_mag, vort=vort, vort_mag=vort_mag, kinetic_energy=kinetic_energy, enstrophy=enstrophy, total_circulation=total_circulation, pressure=p_history, mass_flow_in=mass_flow_in, mass_flow_out=mass_flow_out)
end

#function to create heatmap plot with slider
function create_heatmap_window(field_data, field_name, time, grid, cyl, args)
    screen = GLMakie.Screen()
    GLMakie.set_title!(screen, "$field_name Viewer")
    fig = Figure(resolution = (800, 800), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(screen, fig)

    #creates and configures the figure and axis
    fig[1, 1:2] = Label(fig, "$field_name vs Time", halign=:center, fontsize=28, valign=:top)
    n = size(field_data[1], 1)
    current_idx = Observable(1)
    field_obs = @lift(field_data[$current_idx])
    time_obs = @lift(@sprintf("Time = %.4f s", time[$current_idx]))
    x_range = range(0, grid.L, length=n)
    y_range = range(0, grid.L, length=n)
    ax = Axis(fig[2:8, 1], xlabel="x (m)", ylabel="y (m)",  xlabelsize=20, ylabelsize=20, titlesize=22, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
    hm = heatmap!(ax, x_range, y_range, field_obs, colormap=:viridis)
    Colorbar(fig[2:8, 2], hm, label=field_name, labelsize=18, labelcolor=colorant"#10305F")

    if cyl.radius > 0
        n_cyl = size(field_data[1], 1)
        cyl_x_pix = cyl.center[1] / grid.L * n_cyl
        cyl_y_pix = cyl.center[2] / grid.L * n_cyl
        cyl_r_pix = cyl.radius / grid.L * n_cyl
        poly!(ax, Circle(Point2f(cyl_x_pix, cyl_y_pix), cyl_r_pix), color=:black, strokewidth=2)
    end

    time_slider = Slider(fig[9, 1:2], range=1:length(time), startvalue=1,  color_active=colorant"#10305F", color_active_dimmed=colorant"#5B8DEF", color_inactive=colorant"#8BB1FF", linewidth=10, height=50)

    on(time_slider.value) do idx
        current_idx[] = idx
        ax.title = time_obs[]
    end

    ax.title = time_obs[]
end

#function to create scalar vs time plot
function create_scalar_plot_window(time, data, ylabel, title)
    screen = GLMakie.Screen()
    GLMakie.set_title!(screen, title)
    fig = Figure(size = (800, 500), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(screen, fig)
    fig[1, :] = Label(fig, title, halign=:center, fontsize=28, valign=:top)
    ax = Axis(fig[2:10, :], xlabel="Time (s)", ylabel=ylabel, xlabelsize=20, ylabelsize=20, titlesize=22, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F")
    lines!(ax, time, data, linewidth=3, color=colorant"#5B8DEF")
end

#function to create streamline plot with slider
function create_streamline_window(u_data, v_data, background_data, field_name, time, grid, cyl, args, idx)
    screen = GLMakie.Screen()
    GLMakie.set_title!(screen, "$field_name Streamlines")
    fig = Figure(resolution = (800, 800), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(screen, fig)

    #creates and configures the figure and axis
    fig[1, 1:2] = Label(fig, "$field_name Streamlines", halign=:center, fontsize=28, valign=:top)
    n = size(u_data[idx], 1)
    x_range = range(0, grid.L, length=n)
    y_range = range(0, grid.L, length=n)
    ax = Axis(fig[2:8, 1], xlabel="x (m)", ylabel="y (m)", xlabelsize=20, ylabelsize=20, titlesize=22, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
    current_idx = Observable(idx)
    field_obs = @lift(background_data[$current_idx])
    time_obs = @lift(@sprintf("Time = %.4f s", time[$current_idx]))
    hm = heatmap!(ax, x_range, y_range, field_obs, colormap=:viridis)
    Colorbar(fig[2:8, 2], hm, label=field_name, labelsize=18, labelcolor=colorant"#10305F")

    #makes the streamlines on the field
    streamplot = @lift begin
        nloc = size(u_data[$current_idx], 1)
        step = max(1, nloc ÷ 30)
        xs = range(0, grid.L, length=nloc)[1:step:end]
        ys = range(0, grid.L, length=nloc)[1:step:end]
        u_sub = u_data[$current_idx][1:step:end, 1:step:end]
        v_sub = v_data[$current_idx][1:step:end, 1:step:end]
        (xs, ys, u_sub, v_sub)
    end
    stream_arrows = Ref{Any}(nothing)
    on(streamplot) do tpl
        xs, ys, u_sub, v_sub = tpl
        if stream_arrows[] !== nothing
            delete!(ax, stream_arrows[])
        end
        stream_arrows[] = arrows2d!(ax, xs, ys, u_sub, v_sub; tipwidth=0.02, tiplength=0.03, lengthscale=0.05, color=:white)
    end

    #adds the cylinder if present
    if cyl.radius > 0
        cyl_x_pix = cyl.center[1] / grid.L * n
        cyl_y_pix = cyl.center[2] / grid.L * n
        cyl_r_pix = cyl.radius / grid.L * n
        poly!(ax, Circle(Point2f(cyl_x_pix, cyl_y_pix), cyl_r_pix), color=:black, strokewidth=2)
    end

    #creates and updates the slider
    time_slider = Slider(fig[9, 1:2], range=1:length(time), startvalue=idx, color_active=colorant"#10305F", color_active_dimmed=colorant"#5B8DEF", color_inactive=colorant"#8BB1FF", linewidth=10, height=50)
    on(time_slider.value) do i
        current_idx[] = i
        ax.title = time_obs[]
    end
    ax.title = time_obs[]
end

#function to create quiver plot
function create_quiver_window(u_data, v_data, background_data, field_name, time, grid, cyl, args, idx)
    screen = GLMakie.Screen()
    GLMakie.set_title!(screen, "$field_name Quiver")
    fig = Figure(resolution = (800, 800), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(screen, fig)

    fig[1, 1:2] = Label(fig, "$field_name Quiver", halign=:center, fontsize=28, valign=:top)
    n = size(u_data[idx], 1)
    x_range = range(0, grid.L, length=n)
    y_range = range(0, grid.L, length=n)
    ax = Axis(fig[2:8, 1], xlabel="x (m)", ylabel="y (m)", xlabelsize=20, ylabelsize=20, titlesize=22, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
    current_idx = Observable(idx)
    field_obs = @lift(background_data[$current_idx])
    time_obs = @lift(@sprintf("Time = %.4f s", time[$current_idx]))
    hm = heatmap!(ax, x_range, y_range, field_obs, colormap=:viridis)
    Colorbar(fig[2:8, 2], hm, label=field_name, labelsize=18, labelcolor=colorant"#10305F")

    #makes the quivers on the field
    quiv = @lift begin
        nloc = size(u_data[$current_idx], 1)
        step = max(1, nloc ÷ 20)
        xs = range(0, grid.L, length=nloc)[1:step:end]
        ys = range(0, grid.L, length=nloc)[1:step:end]
        u_sub = u_data[$current_idx][1:step:end, 1:step:end]
        v_sub = v_data[$current_idx][1:step:end, 1:step:end]
        (xs, ys, u_sub, v_sub)
    end
    quiv_arrows = Ref{Any}(nothing)
    on(quiv) do tpl
        xs, ys, u_sub, v_sub = tpl
        if quiv_arrows[] !== nothing
            delete!(ax, quiv_arrows[])
        end
        quiv_arrows[] = arrows2d!(ax, xs, ys, u_sub, v_sub; tipwidth=0.03, tiplength=0.04, lengthscale=0.1, color=:black)
    end

    #adds the cylinder if present
    if cyl.radius > 0
        cyl_x_pix = cyl.center[1] / grid.L * n
        cyl_y_pix = cyl.center[2] / grid.L * n
        cyl_r_pix = cyl.radius / grid.L * n
        poly!(ax, Circle(Point2f(cyl_x_pix, cyl_y_pix), cyl_r_pix), color=:black, strokewidth=2)
    end

    #creates and updates the slider
    time_slider = Slider(fig[9, 1:2], range=1:length(time), startvalue=idx, color_active=colorant"#10305F", color_active_dimmed=colorant"#5B8DEF", color_inactive=colorant"#8BB1FF", linewidth=10, height=50)
    on(time_slider.value) do i
        current_idx[] = i
        ax.title = time_obs[]
    end
    ax.title = time_obs[]
end

#function to export Paraview data
function export_paraview_data(u_history, v_history, p_history, grid, args, outdir)
    n = grid.n
    dx = grid.dx
    steps = length(u_history)
    function write_vtk(step, u, v, p, outdir)
        filename = joinpath(outdir, @sprintf("dns_%05d.vtk", step))
        open(filename, "w") do io
            println(io, "# vtk DataFile Version 3.0")
            println(io, "DNS snapshot step=$(step)")
            println(io, "ASCII")
            println(io, "DATASET STRUCTURED_POINTS")
            println(io, @sprintf("DIMENSIONS %d %d %d", n, n, 1))
            println(io, "ORIGIN 0 0 0")
            println(io, @sprintf("SPACING %.8f %.8f 1.0", dx, dx))
            println(io, @sprintf("POINT_DATA %d", n*n))
            println(io, "SCALARS pressure float 1")
            println(io, "LOOKUP_TABLE default")
            for j in 1:n
                for i in 1:n
                    @printf(io, "%.7e\n", p[i, j])
                end
            end
            println(io, "VECTORS velocity float")
            for j in 1:n
                for i in 1:n
                    @printf(io, "%.7e %.7e 0.0\n", u[i, j], v[i, j])
                end
            end
        end
    end
    for step in 1:steps
        write_vtk(step, u_history[step], v_history[step], p_history[step], outdir)
    end
end

#function to export all graphs 
function export_all_graphs(args, stats, u_history, v_history, p_history, sol, grid, cyl, outdir)
    mkpath(outdir)
    time = stats.time
    nsteps = length(time)

    #function to pick the indices for the graphs
    function pick_indices(n)
        if n >= 5
            return unique(sort([1, Int(clamp(round(n/4), 1, n)), Int(clamp(round(n/2), 1, n)),
                                Int(clamp(round(3n/4), 1, n)), n]))
        elseif n == 4
            return [1,2,3,4]
        elseif n == 3
            return [1,2,3]
        elseif n == 2
            return [1,2]
        else
            return [1]
        end
    end
    idxs = pick_indices(nsteps)

    #function to save the heatmaps
    function save_heatmap(field, name, idx)
        fig = Figure(size=(900,900), backgroundcolor=colorant"#F0F0F0", font="Times New Roman")
        ax = Axis(fig[1,1], xlabel="x (m)", ylabel="y (m)", xlabelsize=18, ylabelsize=18, title="$name @ t=$(round(time[idx]; digits=4))", titlesize=20, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
        n = size(field[1],1)
        x = range(0, grid.L, length=n)
        y = range(0, grid.L, length=n)
        hm = heatmap!(ax, x, y, field[idx], colormap=:viridis)
        Colorbar(fig[1,2], hm, label=name, labelcolor=colorant"#10305F")
        if cyl.radius > 0
            cyl_x = cyl.center[1] / grid.L * n
            cyl_y = cyl.center[2] / grid.L * n
            cyl_r = cyl.radius / grid.L * n
            poly!(ax, Circle(Point2f(cyl_x, cyl_y), cyl_r), color=:black, strokewidth=2)
        end
        save(joinpath(outdir, "$(name)_t$(idx).png"), fig)
    end

    #function to save the streamline plots
    function save_streamline(background, name, idx)
        fig = Figure(size=(900,900), backgroundcolor=colorant"#F0F0F0", font="Times New Roman")
        ax = Axis(fig[1,1], xlabel="x (m)", ylabel="y (m)", xlabelsize=18, ylabelsize=18, title="$name Streamlines @ t=$(round(time[idx]; digits=4))", titlesize=20, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
        n = size(background[1],1)
        x = range(0, grid.L, length=n)
        y = range(0, grid.L, length=n)
        hm = heatmap!(ax, x, y, background[idx], colormap=:viridis)
        Colorbar(fig[1,2], hm, label=name, labelcolor=colorant"#10305F")
        step = max(1, n ÷ 30)
        xs = x[1:step:end]
        ys = y[1:step:end]
        u_sub = u_history[idx][1:step:end, 1:step:end]
        v_sub = v_history[idx][1:step:end, 1:step:end]
        arrows2d!(ax, xs, ys, u_sub, v_sub; tipwidth=0.02, tiplength=0.03, lengthscale=0.05, color=:white)
        if cyl.radius > 0
            cyl_x = cyl.center[1] / grid.L * n
            cyl_y = cyl.center[2] / grid.L * n
            cyl_r = cyl.radius / grid.L * n
            poly!(ax, Circle(Point2f(cyl_x, cyl_y), cyl_r), color=:black, strokewidth=2)
        end
        save(joinpath(outdir, "$(name)_stream_t$(idx).png"), fig)
    end

    #function to save the quiver plots
    function save_quiver(background, name, idx)
        fig = Figure(size=(900,900), backgroundcolor=colorant"#F0F0F0", font="Times New Roman")
        ax = Axis(fig[1,1], xlabel="x (m)", ylabel="y (m)", xlabelsize=18, ylabelsize=18, title="$name Quiver @ t=$(round(time[idx]; digits=4))", titlesize=20, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F", limits=(0, grid.L, 0, grid.L))
        n = size(background[1],1)
        x = range(0, grid.L, length=n)
        y = range(0, grid.L, length=n)
        hm = heatmap!(ax, x, y, background[idx], colormap=:viridis)
        Colorbar(fig[1,2], hm, label=name, labelcolor=colorant"#10305F")
        step = max(1, n ÷ 20)
        xs = x[1:step:end]
        ys = y[1:step:end]
        u_sub = u_history[idx][1:step:end, 1:step:end]
        v_sub = v_history[idx][1:step:end, 1:step:end]
        arrows2d!(ax, xs, ys, u_sub, v_sub; tipwidth=0.03, tiplength=0.04, lengthscale=0.1, color=:black)
        if cyl.radius > 0
            cyl_x = cyl.center[1] / grid.L * n
            cyl_y = cyl.center[2] / grid.L * n
            cyl_r = cyl.radius / grid.L * n
            poly!(ax, Circle(Point2f(cyl_x, cyl_y), cyl_r), color=:black, strokewidth=2)
        end
        save(joinpath(outdir, "$(name)_quiver_t$(idx).png"), fig)
    end

    #function to save the scalar plots
    function save_scalar(time, data, ylabel, title, filename)
        fig = Figure(size=(800,600), backgroundcolor=colorant"#F0F0F0", font="Times New Roman")
        ax = Axis(fig[1,1], xlabel="Time (s)", ylabel=ylabel, xlabelsize=18, ylabelsize=18, titlesize=20, title=title, titlecolor=colorant"#10305F", xlabelcolor=colorant"#10305F", ylabelcolor=colorant"#10305F")
        lines!(ax, time, data, color=colorant"#5B8DEF", linewidth=3)
        save(joinpath(outdir, filename), fig)
    end

    #saves the heatmaps at selected times
    for idx in idxs
        save_heatmap(stats.pressure, "Pressure", idx)
        save_heatmap(stats.vel_mag, "VelocityMagnitude", idx)
        save_heatmap(stats.vort_mag, "VorticityMagnitude", idx)
        save_streamline(stats.vel_mag, "Velocity", idx)
        save_streamline(stats.vort_mag, "Vorticity", idx)
        save_quiver(stats.vel_mag, "Velocity", idx)
    end

    #saves the scalar plots
    save_scalar(time, stats.kinetic_energy, "Kinetic Energy", "Kinetic Energy vs Time", "kinetic_energy.png")
    save_scalar(time, stats.enstrophy, "Enstrophy", "Enstrophy vs Time", "enstrophy.png")
    save_scalar(time, sol.div_history, "Max |div u|", "Divergence vs Time", "divergence.png")
    save_scalar(time, [r[end] for r in sol.residual_history], "Residual", "Pressure Residual vs Time", "residual.png")
    save_scalar(time, stats.total_circulation, "Total Circulation", "Total Circulation vs Time", "circulation.png")
    save_scalar(time, stats.mass_flow_in, "Mass Flow In", "Mass Flow In vs Time", "mass_flow_in.png")
    save_scalar(time, stats.mass_flow_out, "Mass Flow Out", "Mass Flow Out vs Time", "mass_flow_out.png")
end

#function to create visualization window
function create_visualization_window(args, u_history, v_history, p_history, sol, grid, cyl, progress_screen, progress_fig)
    #close progress window
    try
        close(progress_screen)
    catch
    end
    
    #computes the statistics
    dx = grid.dx
    dt = args["dt"]
    stats = compute_statistics(u_history, v_history, p_history, sol.div_history, sol.residual_history, dx, dt)
    
    #creates the visualization window
    vis_screen = GLMakie.Screen()
    GLMakie.set_title!(vis_screen, "2D DNS Simulation - Results")
    vis_fig = Figure(size = (1200, 800), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(vis_screen, vis_fig)
    
    #sets up the grid layout
    for r in 1:5, c in 1:4
        vis_fig[r, c] = Label(vis_fig, "")
    end
    for col in 1:4
        colsize!(vis_fig.layout, col, Relative(0.25))
    end
    for row in 1:5
        rowsize!(vis_fig.layout, row, Relative(0.2))
    end
    colgap!(vis_fig.layout, 40)
    rowgap!(vis_fig.layout, 30)
    
    #creates and configures the gui elemnts
    vis_fig[1, :] = Label(vis_fig, "Simulation Results - Select Visualization", halign=:center, fontsize=32, valign=:top, color=colorant"#10305F")
    plot_buttons = [
        ("Pressure Heatmap", () -> create_heatmap_window(stats.pressure, "Pressure", stats.time, grid, cyl, args)),
        ("Velocity Magnitude Heatmap", () -> create_heatmap_window(stats.vel_mag, "Velocity Magnitude", stats.time, grid, cyl, args)),
        ("Vorticity Heatmap", () -> create_heatmap_window(stats.vort_mag, "Vorticity Magnitude", stats.time, grid, cyl, args)),
        ("Kinetic Energy vs Time", () -> create_scalar_plot_window(stats.time, stats.kinetic_energy, "Kinetic Energy", "Kinetic Energy vs Time")),
        ("Enstrophy vs Time", () -> create_scalar_plot_window(stats.time, stats.enstrophy, "Enstrophy", "Enstrophy vs Time")),
        ("Divergence vs Time", () -> create_scalar_plot_window(stats.time, sol.div_history, "Max |div u|", "Divergence vs Time")),
        ("Residual vs Time", () -> create_scalar_plot_window(stats.time, [r[end] for r in sol.residual_history], "Residual", "Pressure Solver Residual vs Time")),
        ("Mass Flow In vs Time", () -> create_scalar_plot_window(stats.time, stats.mass_flow_in, "Mass Flow In", "Mass Flow In vs Time")),
        ("Mass Flow Out vs Time", () -> create_scalar_plot_window(stats.time, stats.mass_flow_out, "Mass Flow Out", "Mass Flow Out vs Time")),
        ("Velocity Streamlines", () -> create_streamline_window(u_history, v_history, stats.vel_mag, "Velocity", stats.time, grid, cyl, args, length(stats.time))),
        ("Vorticity Streamlines", () -> create_streamline_window(u_history, v_history, stats.vort_mag, "Vorticity", stats.time, grid, cyl, args, length(stats.time))),
        ("Velocity Quiver", () -> create_quiver_window(u_history, v_history, stats.vel_mag, "Velocity", stats.time, grid, cyl, args, length(stats.time))),
    ]
    
    #create buttons in grid
    for (i, (label, func)) in enumerate(plot_buttons)
        row = 2 + Int(fld(i-1, 4))
        col = 1 + ((i-1) % 4)
        btn = Button(vis_fig, label=label)
        vis_fig[row, col] = btn
        on(btn.clicks) do _
            func()
        end
    end
    
    #creates and configures the action buttons
    restart_btn = Button(vis_fig, label="      Restart      ")
    vis_fig[12, 1] = restart_btn
    on(restart_btn.clicks) do _
        close(vis_screen)
        dns_simulation_app()
    end
    exit_btn = Button(vis_fig, label="        Exit        ")
    vis_fig[12, 2] = exit_btn
    on(exit_btn.clicks) do _
        exit()
    end
    export_btn = Button(vis_fig, label="  Export Paraview  ")
    vis_fig[12, 3] = export_btn
    on(export_btn.clicks) do _
        @async begin
            try
                if isdefined(Main, :NativeFileDialog) && isdefined(NativeFileDialog, :pick_folder)
                    outdir = NativeFileDialog.pick_folder()
                else
                    outdir = nothing
                end
                if outdir !== nothing
                    mkpath(outdir)
                    export_paraview_data(u_history, v_history, p_history, grid, args, outdir)
                else
                    outdir = joinpath(pwd(), "paraview_output")
                    mkpath(outdir)
                    export_paraview_data(u_history, v_history, p_history, grid, args, outdir)
                end
            catch
                outdir = joinpath(pwd(), "paraview_output")
                mkpath(outdir)
                export_paraview_data(u_history, v_history, p_history, grid, args, outdir)
            end
        end
    end
    export_graphs_btn = Button(vis_fig, label="  Export Graphs   ")
    vis_fig[12, 4] = export_graphs_btn
    on(export_graphs_btn.clicks) do _
        @async begin
            try
                if isdefined(Main, :NativeFileDialog) && isdefined(NativeFileDialog, :pick_folder)
                    outdir = NativeFileDialog.pick_folder()
                else
                    outdir = nothing
                end
                graphs_dir = if outdir !== nothing
                    joinpath(outdir, "graphs")
                else
                    joinpath(pwd(), "graphs")
                end
                mkpath(graphs_dir)
                export_all_graphs(args, stats, u_history, v_history, p_history, sol, grid, cyl, graphs_dir)
            catch
                graphs_dir = joinpath(pwd(), "graphs")
                mkpath(graphs_dir)
                export_all_graphs(args, stats, u_history, v_history, p_history, sol, grid, cyl, graphs_dir)
            end
        end
    end
end

#creates the function to launch the app
function dns_simulation_app()
    #function that calls the backend
    function run_dns_backend(args, progress_obs, status_label_obs, screen, fig)
        #gets all the parameter for thbe simulation
        L = args["L"]
        n = args["n"]
        grid = DNSBackend.GridParams(n; L=L)
        cyl = DNSBackend.CylinderParams((args["cyl_x"], args["cyl_y"]), args["cyl_radius"])
        fluid = DNSBackend.fluid_from(rho=1.0, nu=args["nu"])
        T = args["dt"] * args["steps"]
        time_params = DNSBackend.TimeParams(T, args["steps"])
        mg_params = DNSBackend.MGParams(cycle_type=:w_cycle, smoother=args["smoother"], sweeps=args["mg_sweeps"], ω=args["mg_omega"], max_cycles=args["solver_max_cycles"], tolerance=args["solver_tol"], initial_guess=:previous, bc=MultigridBackend.default_pressure_bc())
        ode_params = DNSBackend.ODEParams(method=args["ode_method"])
        velocity_bc = DNSBackend.make_velocity_bc(args["Uin"])
        
        #storage for all time steps
        u_history = Vector{Matrix{Float64}}()
        v_history = Vector{Matrix{Float64}}()
        p_history = Vector{Matrix{Float64}}()
        
        #creates progress callback that updates the observable and stores data
        progress_callback = (step, u, v, p, grid) -> begin
            progress = step / args["steps"]
            progress_obs[] = progress
            t_current = (step - 1) * args["dt"]
            status_label_obs[] = @sprintf("Running Simulation: Step %d/%d (t = %.4f s / %.4f s) - %.1f%%", step, args["steps"], t_current, T, progress * 100)
            push!(u_history, copy(u[2:end-1, 2:end-1]))
            push!(v_history, copy(v[2:end-1, 2:end-1]))
            push!(p_history, copy(p[2:end-1, 2:end-1]))
        end
        
        #runs the backend simulation
        try
            sol = DNSBackend.run_dns(grid, cyl, fluid, time_params, mg_params, ode_params; velocity_bc! = velocity_bc, subtract_mean_rhs=true, pressure_solver=args["pressure_solver"], krylov_tol=args["solver_tol"], krylov_max_iters=args["solver_max_cycles"], verbose=false, save_callback=progress_callback)
            progress_obs[] = 1.0
            status_label_obs[] = @sprintf("Simulation Complete! Processing results...")
            create_visualization_window(args, u_history, v_history, p_history, sol, grid, cyl, screen, fig)
        catch e
            status_label_obs[] = "Simulation Failed: $(string(e))"
            println("Error: ", e)
        end
    end

    #creates the screen and the figure
    screen = GLMakie.Screen()
    GLMakie.set_title!(screen, "2D DNS Simulation - Control")
    fig = Figure(size = (1200, 720), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
    display(screen, fig)

    #customizes the grid layout of the figure
    for r in 1:12, c in 1:4
        fig[r, c] = Label(fig, "")
    end
    for col in 1:4
        colsize!(fig.layout, col, Relative(0.25))
    end
    rowsize!(fig.layout, 1, Relative(0.13))
    for r in 2:11
        rowsize!(fig.layout, r, Relative(0.085))
    end
    rowsize!(fig.layout, 11, Relative(0.08))
    rowsize!(fig.layout, 12, Relative(0.1))
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 6)

    #creates a custom theme for the widgets
    custom_theme = Theme(
        Button = (fontsize = 26, buttoncolor = colorant"#5B8DEF", labelcolor = colorant"#10305F", strokewidth = 0, cornerradius = 100, buttoncolor_active = colorant"#ffffff", buttoncolor_hover = colorant"#8BB1FF", labelcolor_hover = colorant"#10305F", labelcolor_active = colorant"#F0F0F0", justification = :center, font="Times New Roman", valign = :bottom, halign = :center),
        Label = (fontsize = 20, color = colorant"#10305F", justification = :right, font="Times New Roman", padding = (2, 6, 2, 6)),
        Textbox = (fontsize = 22, textcolor = colorant"#10305F", justification = :left, borderwidth = 0, textcolor_placeholder = colorant"#10305F", placeholder = "Enter..", cursorcolor = colorant"#10305F", font="Times New Roman", padding = (4, 6, 4, 6)),
        Menu = (fontsize = 22, textcolor = colorant"#10305F", font = "Times New Roman", cell_color_active = colorant"#F0F0F0", cell_color_inactive_even = colorant"#F0F0F0", cell_color_inactive_odd = colorant"#F0F0F0", cell_color_hover = colorant"#5B8DEF", strokewidth = 0, direction = :down, padding = (4, 6, 4, 6))
    )
    set_theme!(custom_theme)

    #creates the main label at the top
    main_label_text = Observable("2D DNS Simulation - Enter Parameters")
    main_label = Label(fig, main_label_text, halign = :center, fontsize = 34, valign = :top)
    fig[1, :] = main_label

    #defines the parameters that the user must input
    param_names = ["Inlet Velocity (m/s):", "Kinematic Viscosity ν (m²/s):", "Time Step (dt):", "Number of Steps:", "Cells Per Side (interior):", "Cylinder Radius:", "Cylinder Center X:", "Cylinder Center Y:", "MG Sweeps:", "MG Tolerance:", "MG ω:", "MG Max Cycles:"]

    #creates the labels that hold the parameter names and the entries to hold them
    entries = []
    function add_field(row, col_label, label_text, default_value = nothing)
        fig[row, col_label] = Label(fig, label_text, halign = :right, justification = :right)
        box = Textbox(fig, halign = :left)
        if default_value !== nothing
            Makie.set!(box, string(default_value))
        end
        fig[row, col_label + 1] = box
        push!(entries, box)
    end
    add_field(2, 1, "Domain Length L (m):", 1.0)
    add_field(2, 3, "Inlet Velocity U (m/s):", 1.0)
    add_field(3, 1, "Kinematic Viscosity ν (m²/s):", 0.002)
    add_field(3, 3, "Time Step dt (s):", 3e-3)
    add_field(4, 1, "Number of time Steps N:", 100)
    add_field(4, 3, "Grid Size n:", 128)
    add_field(5, 1, "Cylinder Radius:", 0.3)
    add_field(5, 3, "Cylinder Center x (m):", 0.5)
    add_field(6, 1, "Cylinder Center y (m):", 0.5)
    add_field(6, 3, "Multigrid Sweeps:", 7)
    add_field(7, 1, "Solver Tolerance:", 1e-6)
    add_field(7, 3, "Multigrid ω:", 0.7)
    add_field(8, 1, "Solver Max Cycles:", 100)

    #defines the momentum ode and pressure pde solver choices
    ode_options = Dict(
        "RK1" => :rk1, "RK2" => :rk2, "RK3" => :rk3, "RK4" => :rk4, "RK5" => :rk5,
        "SSPRK2" => :ssprk2, "SSPRK3" => :ssprk3,
        "GL2" => :gauss2, "GL3" => :gauss3, "GL4" => :gauss4, "GL5" => :gauss5,
        "Radau2" => :radau2, "Radau3" => :radau3, "Radau4" => :radau4, "Radau5" => :radau5,
        "Lobatto2" => :lobatto2, "Lobatto3" => :lobatto3, "Lobatto4" => :lobatto4, "Lobatto5" => :lobatto5,
        "SDIRK2" => :sdirk2, "SDIRK3" => :sdirk3, "SDIRK4" => :sdirk4)
    pressure_options = Dict("Multigrid" => :mg, "PCG" => :pcg, "PFGMRES" => :fgmres)
    smoother_options = Dict("Weighted Jacobi" => :weighted_jacobi, "RBGS" => :rbgs)

    #creates the dropdown menus with labels
    fig[8, 3] = Label(fig, "ODE Solver:", halign = :right, justification = :right)
    ode_menu = Menu(fig[9, 2], options = collect(keys(ode_options)), halign = :left, default = "SSPRK2")
    fig[9, 1] = Label(fig, "Pressure Solver:", halign = :right, justification = :right)
    pressure_menu = Menu(fig[9, 4], options = collect(keys(pressure_options)), halign = :left, default = "PFGMRES")
    fig[9, 3] = Label(fig, "Smoother:", halign = :right, justification = :right)
    smoother_menu = Menu(fig[8, 4], options = collect(keys(smoother_options)), halign = :left, default = "Weighted Jacobi")

    #defines and creates the buttons for the preset viscosities
    viscosity_presets = Dict("       Water        " => 1e-6, "        Air        " => 1.5e-5, "        Oil        " => 1e-4, "       Honey       " => 1e-2, "      Glycerin      " => 1.2e-3, "       Blood       "  => 3.5e-6, "      Gasoline      " => 5e-7)
    preset_buttons = Dict{String, Button}()
    preset_names = collect(keys(viscosity_presets))
    for (i, name) in enumerate(preset_names)
        btn = Button(fig, label = name)
        row = 11 + Int(fld(i-1, 4))
        col = 1 + ((i-1) % 4)
        fig[row, col] = btn
        preset_buttons[name] = btn
    end

    #function that applies the preset viscosities to the correct values
    function apply_viscosity!(entry_vec, material_name)
        Makie.set!(entry_vec[3], string(viscosity_presets[material_name]))
    end
    for name in preset_names
        on(preset_buttons[name].clicks) do _
            apply_viscosity!(entries, name)
        end
    end

    #reads the textbox as a number with fallback
    read_float(box) = parse(Float64, box.displayed_string[])
    read_int(box) = parse(Int, box.displayed_string[])

    #function that confirms that all the inputs are valid and begins the calculation
    function begin_run!(entries, ode_menu, pressure_menu, smoother_menu)
        args = Dict{String, Any}()
        try
            L = read_float(entries[1])
            Uin = read_float(entries[2])
            nu = read_float(entries[3])
            dt = read_float(entries[4])
            steps = read_int(entries[5])
            cells = read_int(entries[6])
            cyl_r = read_float(entries[7])
            cyl_x = read_float(entries[8])
            cyl_y = read_float(entries[9])
            mg_sweeps = read_int(entries[10])
            solver_tol = read_float(entries[11])
            mg_omega = read_float(entries[12])
            solver_max_cycles = read_int(entries[13])

            #checks for positivity and basic validity
            if Uin <= 0 || nu <= 0 || dt <= 0 || steps < 1 || cells < 2 || mg_sweeps < 1 || solver_tol <= 0 || mg_omega <= 0 || solver_max_cycles < 1
                main_label_text[] = "2D DNS Simulation - Inputs Must Be Positive and Valid"
                return nothing
            end

            #checks for the domain and cylinder to be valid
            if cyl_r < 0
                main_label_text[] = "2D DNS Simulation - Cylinder Radius Must Be ≥ 0"
                return nothing
            end
            if !(0 < cyl_x < L) || !(0 < cyl_y < L)
                main_label_text[] = "2D DNS Simulation - Cylinder Center Must Be Within Domain"
                return nothing
            end
            if (cyl_x + cyl_r >= L) || (cyl_x - cyl_r <= 0) || (cyl_y + cyl_r >= L) || (cyl_y - cyl_r <= 0)
                main_label_text[] = "2D DNS Simulation - Cylinder Must Fit Inside Domain"
                return nothing
            end
            n = cells

            #collects the menu selections
            ode_method = ode_options[ode_menu.selection[]]
            pressure_solver = pressure_options[pressure_menu.selection[]]
            smoother = smoother_options[smoother_menu.selection[]]

            #fills all the arguments
            args["L"] = L
            args["Uin"] = Uin
            args["nu"] = nu
            args["dt"] = dt
            args["steps"] = steps
            args["cells"] = cells
            args["n"] = n
            args["cyl_radius"] = cyl_r
            args["cyl_x"] = cyl_x
            args["cyl_y"] = cyl_y
            args["ode_method"] = ode_method
            args["pressure_solver"] = pressure_solver
            args["smoother"] = smoother
            args["mg_sweeps"] = mg_sweeps
            args["solver_tol"] = solver_tol
            args["mg_omega"] = mg_omega
            args["solver_max_cycles"] = solver_max_cycles
        #retusn an error message if the inputs are not valid
        catch
            main_label_text[] = "2D DNS Simulation - Please Ensure Inputs Are Numbers"
            return nothing
        end

        #closes the old window and creates a new one for progress UI
        try
            close(screen)
        catch
        end
        progress_screen = GLMakie.Screen()
        GLMakie.set_title!(progress_screen, "2D DNS Simulation - Running")
        progress_fig = Figure(size = (1200, 400), backgroundcolor = colorant"#F0F0F0", font="Times New Roman")
        display(progress_screen, progress_fig)
        
        #creates observables for progress tracking
        progress_obs = Observable(0.0)
        status_label_obs = Observable("Initializing Simulation...")
        
        #creates new layout for progress screen
        progress_fig[1, :] = Label(progress_fig, status_label_obs, halign = :center, fontsize = 28, valign = :center)
        ax = Axis(progress_fig[2:4, :], limits = (0, 1, 0, 1),  xgridvisible = false, ygridvisible = false,  xminorgridvisible = false, yminorgridvisible = false,  bottomspinecolor = :transparent, topspinecolor = :transparent, leftspinecolor = :transparent, rightspinecolor = :transparent, xtickcolor = :transparent, ytickcolor = :transparent, xticklabelcolor = :transparent, yticklabelcolor = :transparent, backgroundcolor = colorant"#BCD3F4") 
        hidexdecorations!(ax, grid = false, ticks = false, ticklabels = false, label = false)
        hideydecorations!(ax, grid = false, ticks = false, ticklabels = false, label = false)
        progress_rect = @lift(Rect(0, 0, $progress_obs, 1))
        poly!(ax, progress_rect, color = colorant"#5B8DEF", strokewidth = 0)
        
        #runs the backend
        @async begin
            run_dns_backend(args, progress_obs, status_label_obs, progress_screen, progress_fig)
        end
    end

    #creates the button that calls the function to begin the calculation
    start_btn = Button(fig, label = " Begin Simulation ")
    fig[12, 4] = start_btn
    on(start_btn.clicks) do _
        begin_run!(entries, ode_menu, pressure_menu, smoother_menu)
    end
end

#calls the function to begin the program
dns_simulation_app()
