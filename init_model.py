from model import MPC_Parameters, Robot_Model
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np


def init_model(x0):
    ocp = AcadosOcp()
    params = MPC_Parameters()
    model, f_expl, nx, nu = Robot_Model()
    ocp.model = model

    Q_mat = 2 * np.diag([5, 5, 0.1])
    R_mat = 2 * np.diag([0.1, 0.01])
    N = 50
    ocp.dims.N = N
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (model.x - params.x_ref).T @ Q_mat @ (
        model.x - params.x_ref
    ) + (model.u - params.u_ref).T @ R_mat @ (model.u - params.u_ref)
    ocp.model.cost_expr_ext_cost_e = (
        (model.x - params.x_ref).T @ Q_mat @ (model.x - params.x_ref)
    )
    Fmax = 80
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_max_iter = 100

    ocp.solver_options.tf = 0.41
    ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    u0 = ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False)

    # for i in range(N):
    #     simX[i, :] = ocp_solver.get(i, "x")
    #     simU[i, :] = ocp_solver.get(i, "u")
    # simX[N, :] = ocp_solver.get(N, "x")

    return u0
