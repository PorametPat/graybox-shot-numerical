import jax
import jax.numpy as jnp
from functools import partial
from flax import linen as nn
from flax.typing import VariableDict
import optax
import typing
from inspeqtor.experimental.optimize import DataBundled, HistoryEntryV3
from inspeqtor.experimental.models.linen import create_step
from inspeqtor.experimental.utils import dataloader
from inspeqtor.experimental.data import QubitInformation
from inspeqtor.experimental.physics import (
    signal_func_v5,
    explicit_auto_rotating_frame_hamiltonian,
    make_trotterization_solver,
)
from inspeqtor.experimental.predefined import transmon_hamiltonian
from inspeqtor.experimental.constant import Z
from inspeqtor.v2.control import get_envelope_transformer
from inspeqtor.v2.predefined import get_drag_pulse_v2_sequence
from inspeqtor.v2.utils import SyntheticDataModel

def get_predefined_data_model_m1_v0(
    detune: float = 0.0005, get_envelope_transformer=get_envelope_transformer
):
    
    FREQUENCY = 5.0
    
    dt = 2 / 9
    real_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=FREQUENCY,
        drive_strength=0.1,
    )

    characterized_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=FREQUENCY * (1 + detune),
        drive_strength=0.1,
    )

    control_seq = get_drag_pulse_v2_sequence(
        qubit_info_drive_strength=characterized_qubit_info.drive_strength,
        min_beta=-10.0,
        max_beta=0,
        dt=dt,
    )

    signal_fn = signal_func_v5(
        get_envelope=get_envelope_transformer(control_seq),
        drive_frequency=characterized_qubit_info.frequency,
        dt=dt,
    )
    hamiltonian = partial(
        transmon_hamiltonian, qubit_info=real_qubit_info, signal=signal_fn
    )
    frame = (jnp.pi * characterized_qubit_info.frequency) * Z
    hamiltonian = explicit_auto_rotating_frame_hamiltonian(hamiltonian, frame=frame)

    TROTTER_STEPS = 10_000

    solver = make_trotterization_solver(
        hamiltonian=hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    ideal_hamiltonian = partial(
        transmon_hamiltonian,
        qubit_info=characterized_qubit_info,
        signal=signal_fn,  # Already used the characterized_qubit
    )
    ideal_hamiltonian = explicit_auto_rotating_frame_hamiltonian(
        ideal_hamiltonian, frame=frame
    )

    whitebox = make_trotterization_solver(
        hamiltonian=ideal_hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    return SyntheticDataModel(
        control_sequence=control_seq,
        qubit_information=characterized_qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=hamiltonian,
        solver=solver,
        quantum_device=None,
        whitebox=whitebox,
    )


def get_predefined_data_model_m1(
    detune: float = 0.0005, get_envelope_transformer=get_envelope_transformer
):
    
    FREQUENCY = 5.0
    
    dt = 2 / 9
    real_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=FREQUENCY + detune,
        drive_strength=0.1,
    )

    characterized_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=FREQUENCY,
        drive_strength=0.1,
    )

    control_seq = get_drag_pulse_v2_sequence(
        qubit_info_drive_strength=characterized_qubit_info.drive_strength,
        min_beta=0.,
        max_beta=10.,
        dt=dt,
    )

    signal_fn = signal_func_v5(
        get_envelope=get_envelope_transformer(control_seq),
        drive_frequency=characterized_qubit_info.frequency,
        dt=dt,
    )
    hamiltonian = partial(
        transmon_hamiltonian, qubit_info=real_qubit_info, signal=signal_fn
    )
    frame = (jnp.pi * characterized_qubit_info.frequency) * Z
    hamiltonian = explicit_auto_rotating_frame_hamiltonian(hamiltonian, frame=frame)

    TROTTER_STEPS = 10_000

    solver = make_trotterization_solver(
        hamiltonian=hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    ideal_hamiltonian = partial(
        transmon_hamiltonian,
        qubit_info=characterized_qubit_info,
        signal=signal_fn,  # Already used the characterized_qubit
    )
    ideal_hamiltonian = explicit_auto_rotating_frame_hamiltonian(
        ideal_hamiltonian, frame=frame
    )

    whitebox = make_trotterization_solver(
        hamiltonian=ideal_hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    return SyntheticDataModel(
        control_sequence=control_seq,
        qubit_information=characterized_qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=hamiltonian,
        solver=solver,
        quantum_device=None,
        whitebox=whitebox,
    )


def bound(expected_loss, shots):
    return jnp.nan_to_num(jnp.sqrt((3 / 4) * (expected_loss - (2 / (3 * shots)))))


def train_model(
    # Random key
    key: jnp.ndarray,
    # Data
    train_data: DataBundled,
    val_data: DataBundled,
    test_data: DataBundled,
    # Model to be used for training
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    # Loss function to be used
    loss_fn: typing.Callable,
    # Callbacks to be used
    callbacks: list[typing.Callable] = [],
    early_stop_signal=lambda x, y, z: False,
    # Number of epochs
    NUM_EPOCH: int = 1_000,
    # Optional state
    model_params: VariableDict | None = None,
    opt_state: optax.OptState | None = None,
):
    """Train the BlackBox model

    Examples:
        >>> # The number of epochs break down
        ... NUM_EPOCH = 150
        ... # Total number of iterations as 90% of data is used for training
        ... # 10% of the data is used for testing
        ... total_iterations = 9 * NUM_EPOCH
        ... # The step for optimizer if set to 8 * NUM_EPOCH (should be less than total_iterations)
        ... step_for_optimizer = 8 * NUM_EPOCH
        ... optimizer = get_default_optimizer(step_for_optimizer)
        ... # The warmup steps for the optimizer
        ... warmup_steps = 0.1 * step_for_optimizer
        ... # The cool down steps for the optimizer
        ... cool_down_steps = total_iterations - step_for_optimizer
        ... total_iterations, step_for_optimizer, warmup_steps, cool_down_steps

    Args:
        key (jnp.ndarray): Random key
        model (nn.Module): The model to be used for training
        optimizer (optax.GradientTransformation): The optimizer to be used for training
        loss_fn (typing.Callable): The loss function to be used for training
        callbacks (list[typing.Callable], optional): list of callback functions. Defaults to [].
        NUM_EPOCH (int, optional): The number of epochs. Defaults to 1_000.

    Returns:
        tuple: The model parameters, optimizer state, and the histories
    """

    key, loader_key, init_key = jax.random.split(key, 3)

    train_p, train_u, train_ex = (
        train_data.control_params,
        train_data.unitaries,
        train_data.observables,
    )
    val_p, val_u, val_ex = (
        val_data.control_params,
        val_data.unitaries,
        val_data.observables,
    )
    test_p, test_u, test_ex = (
        test_data.control_params,
        test_data.unitaries,
        test_data.observables,
    )

    BATCH_SIZE = val_p.shape[0]

    if model_params is None:
        # Initialize the model parameters if it is None
        model_params = model.init(init_key, train_p[0])

    if opt_state is None:
        # Initalize the optimizer state if it is None
        opt_state = optimizer.init(model_params)

    # histories: list[dict[str, typing.Any]] = []
    histories: list[HistoryEntryV3] = []

    train_step, eval_step = create_step(
        optimizer=optimizer, loss_fn=loss_fn, has_aux=True
    )

    for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (train_p, train_u, train_ex),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
        model_params, opt_state, (loss, aux) = train_step(
            model_params, opt_state, batch_p, batch_u, batch_ex
        )

        histories.append(HistoryEntryV3(step=step, loss=loss, loop="train", aux=aux))

        if is_last_batch:
            # Validation
            (val_loss, aux) = eval_step(model_params, val_p, val_u, val_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=val_loss, loop="val", aux=aux)
            )

            # Testing
            (test_loss, aux) = eval_step(model_params, test_p, test_u, test_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=test_loss, loop="test", aux=aux)
            )

            for callback in callbacks:
                callback(model_params, opt_state, histories)

            if early_stop_signal(model_params, opt_state, histories):
                break

    return model_params, opt_state, histories

colors = {"blue": "#6366f1", "red": "#f43f5e", "orange": "#f97316", "gray": "gray"}