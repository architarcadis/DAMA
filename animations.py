import random
import html
import streamlit as st

# Constants for consistent styling
ARCADIS_ORANGE = "238, 114, 3"  # RGB values
ARCADIS_BLUE = "#1e3c70"
ARCADIS_GREEN = "#228c22"
ARCADIS_RED = "#c82536"

# Custom loading animations for data processing
def loading_animation(message="Processing", animation_type="pulse", key=None):
    """
    Display an elegant loading animation with custom styling
    
    Parameters:
    - message: Text to display during loading
    - animation_type: Type of animation (pulse, bounce, grow, dots, progress)
    - key: Unique key for the component
    """
    if key is None:
        key = f"loading_{random.randint(1000, 9999)}"
    
    # Different animation types
    if animation_type == "pulse":
        # Pulsing circle with Arcadis orange
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="display: inline-block; position: relative; width: 80px; height: 80px;">
                    <div style="position: absolute; width: 40px; height: 40px; background: rgb({ARCADIS_ORANGE}); 
                            border-radius: 50%; left: 20px; top: 20px; animation: pulse 1.5s ease-in-out infinite;">
                    </div>
                </div>
                <p style="margin-top: 10px; color: #555;">{html.escape(message)}</p>
            </div>
            <style>
                @keyframes pulse {{
                    0% {{ transform: scale(0.8); opacity: 0.7; }}
                    50% {{ transform: scale(1.2); opacity: 1; }}
                    100% {{ transform: scale(0.8); opacity: 0.7; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif animation_type == "bounce":
        # Bouncing dots
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="display: inline-block; position: relative; width: 80px; height: 80px;">
                    <div style="position: absolute; width: 15px; height: 15px; background: rgb({ARCADIS_ORANGE}); 
                            border-radius: 50%; left: 10px; top: 32px; animation: bounce 1.5s ease-in-out infinite;">
                    </div>
                    <div style="position: absolute; width: 15px; height: 15px; background: rgb({ARCADIS_ORANGE}); 
                            border-radius: 50%; left: 32px; top: 32px; animation: bounce 1.5s ease-in-out 0.2s infinite;">
                    </div>
                    <div style="position: absolute; width: 15px; height: 15px; background: rgb({ARCADIS_ORANGE}); 
                            border-radius: 50%; left: 54px; top: 32px; animation: bounce 1.5s ease-in-out 0.4s infinite;">
                    </div>
                </div>
                <p style="margin-top: 10px; color: #555;">{html.escape(message)}</p>
            </div>
            <style>
                @keyframes bounce {{
                    0%, 100% {{ transform: translateY(0); }}
                    50% {{ transform: translateY(-20px); }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif animation_type == "grow":
        # Growing bar with progress animation
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="display: inline-block; position: relative; width: 150px; height: 8px; background-color: #eee; border-radius: 4px; overflow: hidden;">
                    <div style="position: absolute; height: 100%; width: 30%; background: rgb({ARCADIS_ORANGE}); 
                         border-radius: 4px; animation: grow 2s ease-in-out infinite;">
                    </div>
                </div>
                <p style="margin-top: 10px; color: #555;">{html.escape(message)}</p>
            </div>
            <style>
                @keyframes grow {{
                    0% {{ width: 0%; }}
                    50% {{ width: 100%; }}
                    100% {{ width: 0%; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif animation_type == "dots":
        # Loading dots
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <p style="color: #555; font-size: 16px;">
                    {html.escape(message)}<span class="dots-loader"></span>
                </p>
            </div>
            <style>
                .dots-loader:after {{
                    content: '.';
                    animation: dots 1.5s steps(3, end) infinite;
                }}
                
                @keyframes dots {{
                    0%, 20% {{ content: '.'; }}
                    40% {{ content: '..'; }}
                    60%, 100% {{ content: '...'; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif animation_type == "progress":
        # Progress meter with stages
        # Get a random progress value between 10-90% to simulate work in progress
        progress = random.randint(10, 90)
        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="display: inline-block; position: relative; width: 200px; height: 6px; background-color: #eee; border-radius: 3px; overflow: hidden;">
                    <div style="position: absolute; height: 100%; width: {progress}%; background: rgb({ARCADIS_ORANGE}); 
                         border-radius: 3px; transition: width 0.5s ease;">
                    </div>
                </div>
                <p style="margin-top: 10px; color: #555;">{html.escape(message)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    return key

# Data processing animation with stages
def processing_animation_with_stages(stage, total_stages, stage_name, description="", key=None):
    """
    Show a sophisticated processing animation with stages
    
    Parameters:
    - stage: Current stage number
    - total_stages: Total number of stages
    - stage_name: Name of current stage
    - description: Description of current process
    """
    if key is None:
        key = f"process_{random.randint(1000, 9999)}"
    
    # Calculate progress percentage
    progress_pct = min(100, int((stage / total_stages) * 100))
    
    # Define stage colors
    completed_color = f"rgb({ARCADIS_ORANGE})"
    current_color = ARCADIS_BLUE
    future_color = "#dddddd"
    
    # Create HTML for stage indicators
    stages_html = ""
    for i in range(1, total_stages + 1):
        if i < stage:
            # Completed stage
            stage_color = completed_color
            checkmark = "✓"
            stage_class = "completed"
        elif i == stage:
            # Current stage
            stage_color = current_color
            checkmark = f"{i}"
            stage_class = "current"
        else:
            # Future stage
            stage_color = future_color
            checkmark = f"{i}"
            stage_class = "future"
        
        stages_html += f"""
        <div class="stage-item {stage_class}">
            <div class="stage-circle" style="background-color: {stage_color};">{checkmark}</div>
            <div class="stage-line" style="background-color: {stage_color if i < total_stages else 'transparent'};"></div>
        </div>
        """
    
    # Create animation HTML
    st.markdown(
        f"""
        <div class="processing-animation">
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress_pct}%;"></div>
            </div>
            <div class="stage-name">{html.escape(stage_name)}</div>
            <div class="stage-description">{html.escape(description)}</div>
            <div class="stages-container">
                {stages_html}
            </div>
        </div>
        
        <style>
        .processing-animation {{
            margin: 20px auto;
            padding: 15px;
            max-width: 600px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .progress-container {{
            height: 8px;
            background-color: #eeeeee;
            border-radius: 4px;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background-color: rgb({ARCADIS_ORANGE});
            border-radius: 4px;
            transition: width 0.5s ease-in-out;
        }}
        
        .stage-name {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
            color: {current_color};
        }}
        
        .stage-description {{
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        
        .stages-container {{
            display: flex;
            justify-content: space-between;
            position: relative;
            padding: 10px 0;
        }}
        
        .stage-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            z-index: 1;
            flex: 1;
        }}
        
        .stage-circle {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            margin-bottom: 5px;
        }}
        
        .stage-line {{
            height: 3px;
            width: 100%;
            position: absolute;
            top: 15px;
            left: 50%;
            z-index: -1;
        }}
        
        .completed .stage-circle {{
            animation: pulse_complete 2s infinite;
        }}
        
        .current .stage-circle {{
            animation: pulse_current 2s infinite;
        }}
        
        @keyframes pulse_complete {{
            0% {{ box-shadow: 0 0 0 0 rgba({ARCADIS_ORANGE}, 0.5); }}
            70% {{ box-shadow: 0 0 0 10px rgba({ARCADIS_ORANGE}, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba({ARCADIS_ORANGE}, 0); }}
        }}
        
        @keyframes pulse_current {{
            0% {{ box-shadow: 0 0 0 0 rgba(30, 60, 112, 0.5); }}
            70% {{ box-shadow: 0 0 0 10px rgba(30, 60, 112, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(30, 60, 112, 0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    return key

# Function to create card with loading animation
def loading_card(title, message, animation_type="pulse"):
    """
    Create a card with loading animation, title and message
    
    Parameters:
    - title: Card title
    - message: Description message
    - animation_type: Type of animation to display
    """
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <h3 style="color: {ARCADIS_BLUE}; margin-top: 0;">{html.escape(title)}</h3>
            <p style="color: #666; margin-bottom: 15px;">{html.escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    loading_animation("Processing...", animation_type)