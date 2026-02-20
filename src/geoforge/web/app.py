"""Gradio web UI for GeoForge."""

from __future__ import annotations

import gradio as gr

from geoforge.web.handlers import (
    format_spec_markdown,
    get_provider_choices,
    run_pipeline,
)


async def _handle_generate(prompt: str, provider: str):
    """Handle the generate button click."""
    if not prompt or not prompt.strip():
        return (
            "Please enter a geometry description.",  # status
            None,  # code
            None,  # preview image
            "",  # validation
            "",  # spec info
            None,  # py download
            None,  # gds download
            None,  # oas download
        )

    result = await run_pipeline(prompt.strip(), provider)

    if result.error:
        return (
            f"Error: {result.error}",
            None,
            None,
            "",
            "",
            None,
            None,
            None,
        )

    # Format outputs
    status = "Generation complete!" if result.success else "Generation failed."
    code = result.code or ""
    preview = str(result.preview_path) if result.preview_path else None
    validation = result.validation_markdown or ""
    spec_info = format_spec_markdown(result.spec) if result.spec else ""

    py_file = str(result.py_path) if result.py_path else None
    gds_file = str(result.gds_path) if result.gds_path else None
    oas_file = str(result.oas_path) if result.oas_path else None

    return status, code, preview, validation, spec_info, py_file, gds_file, oas_file


def create_app() -> gr.Blocks:
    """Create the Gradio Blocks app.

    Returns:
        Configured gr.Blocks instance (not launched).
    """
    providers = get_provider_choices()

    with gr.Blocks(
        title="GeoForge - Semiconductor Geometry Generator",
    ) as app:
        gr.Markdown("# GeoForge\n**Natural language semiconductor geometry generator**")

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Geometry Description",
                    placeholder="e.g., Create a 3x3 via array with 2um pitch on metal1 and metal2",
                    lines=3,
                )
            with gr.Column(scale=1):
                provider_dropdown = gr.Dropdown(
                    choices=providers,
                    value=providers[0] if providers else "ollama",
                    label="LLM Provider",
                )
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Generated Code"):
                code_output = gr.Code(language="python", label="GDSFactory Code")

            with gr.Tab("Layout Preview"):
                preview_image = gr.Image(label="Layout Preview", type="filepath")

            with gr.Tab("Validation"):
                validation_output = gr.Markdown(label="Validation Results")

            with gr.Tab("Geometry Spec"):
                spec_output = gr.Markdown(label="Extracted Specification")

        with gr.Row():
            py_download = gr.File(label="Python Code", interactive=False)
            gds_download = gr.File(label="GDS File", interactive=False)
            oas_download = gr.File(label="OAS File", interactive=False)

        # Wire up the generate button
        generate_btn.click(
            fn=_handle_generate,
            inputs=[prompt_input, provider_dropdown],
            outputs=[
                status_box,
                code_output,
                preview_image,
                validation_output,
                spec_output,
                py_download,
                gds_download,
                oas_download,
            ],
        )

        # Also trigger on Enter in the prompt box
        prompt_input.submit(
            fn=_handle_generate,
            inputs=[prompt_input, provider_dropdown],
            outputs=[
                status_box,
                code_output,
                preview_image,
                validation_output,
                spec_output,
                py_download,
                gds_download,
                oas_download,
            ],
        )

    return app


def launch(
    host: str = "0.0.0.0",  # nosec B104
    port: int = 7860,
    share: bool = False,
    **kwargs,
) -> None:
    """Create and launch the Gradio app.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        share: Whether to create a public Gradio share link.
        **kwargs: Additional arguments passed to gr.Blocks.launch().
    """
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
        **kwargs,
    )
