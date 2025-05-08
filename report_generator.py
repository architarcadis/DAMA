import io
import time
import json
import tempfile
import os
from datetime import datetime
import plotly.io as pio
from fpdf import FPDF
import base64
from PIL import Image
import numpy as np

class DataQualityReport(FPDF):
    """Custom PDF class for data quality reports"""
    
    def __init__(self, title="Data Quality Assessment Report", organization="", *args, **kwargs):
        # Initialize FPDF without custom font cache settings
        super().__init__(*args, **kwargs)
        
        # Safely encode title and organization to latin-1, replacing unsupported characters
        try:
            self.title = title.encode('latin-1', errors='replace').decode('latin-1')
        except (UnicodeError, AttributeError):
            self.title = "Data Quality Assessment Report"
            
        try:
            self.organization = organization.encode('latin-1', errors='replace').decode('latin-1')
        except (UnicodeError, AttributeError):
            self.organization = ""
            
        self.set_auto_page_break(auto=True, margin=15)
        # Use standard fonts instead of custom DejaVu
        self._current_section = ""
        
    def header(self):
        """Add report header to each page"""
        # Logo (use text logo instead of image)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        self.cell(0, 8, "DATA QUALITY ASSESSMENT", 0, 1, 'L')
        
        # Header line
        self.set_draw_color(238, 114, 3)  # Arcadis orange
        self.line(10, 10, 200, 10)
        
        # Title
        if self._current_section:
            self.set_font('Helvetica', '', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, self._current_section, 0, 1, 'R')
        
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(128, 128, 128)
        
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        # Date
        self.cell(0, 10, datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')
    
    def chapter_title(self, title):
        """Add chapter title"""
        safe_title = self._encode_text(title)
        self._current_section = safe_title
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        self.cell(0, 10, safe_title, 0, 1, 'L')
        self.ln(4)
    
    def section_title(self, title):
        """Add section title"""
        safe_title = self._encode_text(title)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        self.cell(0, 8, safe_title, 0, 1, 'L')
        self.ln(2)
    
    def sub_section_title(self, title):
        """Add sub section title"""
        safe_title = self._encode_text(title)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(150, 70, 20)  # Darker orange for sub-sections
        self.cell(0, 6, safe_title, 0, 1, 'L')
        self.ln(2)
    
    def _encode_text(self, text):
        """Safely encode text to latin-1 to prevent Unicode errors"""
        if text is None:
            return ""
        try:
            # Encode to latin-1, replacing unsupported characters with '?'
            return text.encode('latin-1', errors='replace').decode('latin-1')
        except (UnicodeError, AttributeError):
            # If any error occurs, return a safe empty string
            return ""
    
    def body_text(self, text):
        """Add body text"""
        # Convert to latin-1 safe text
        safe_text = self._encode_text(text)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, safe_text)
        self.ln(2)
    
    def info_text(self, text):
        """Add informational text (slightly smaller)"""
        # Convert to latin-1 safe text
        safe_text = self._encode_text(text)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, safe_text)
        self.ln(2)
        
    def highlight_box(self, text, background_color=(248, 249, 250), border_color=(238, 114, 3), width=180):
        """Add a highlighted box with important text or insights"""
        # Convert text to latin-1 safe
        safe_text = self._encode_text(text)
        
        # Save current position and colors
        current_x = self.x
        current_y = self.y
        current_fill = self.fill_color
        current_draw = self.draw_color
        current_text = self.text_color
        
        # Set colors for the box with error handling
        try:
            # Check if background_color is a tuple with at least 3 elements
            if isinstance(background_color, tuple) and len(background_color) >= 3:
                r, g, b = background_color[:3]  # Only take the first three elements
            else:
                # Fallback if color is not valid
                r, g, b = 248, 249, 250  # Default light gray
            self.set_fill_color(int(r), int(g), int(b))
            
            # Check if border_color is a tuple with at least 3 elements
            if isinstance(border_color, tuple) and len(border_color) >= 3:
                r, g, b = border_color[:3]  # Only take the first three elements
            else:
                # Fallback if color is not valid
                r, g, b = 238, 114, 3  # Default orange
            self.set_draw_color(int(r), int(g), int(b))
        except Exception:
            # If any error occurs, use defaults
            self.set_fill_color(248, 249, 250)  # Light gray
            self.set_draw_color(238, 114, 3)  # Orange
        
        # Draw the box
        self.rect(current_x, current_y, width, 15, 'FD')
        
        # Add the text
        self.set_xy(current_x + 5, current_y + 4)
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(20, 20, 20)
        self.multi_cell(width - 10, 4, safe_text)
        
        # Calculate new Y position
        new_y = current_y + 15
        
        # Restore colors and set new position with error handling
        try:
            # Handle current_fill
            if isinstance(current_fill, tuple) and len(current_fill) >= 3:
                r, g, b = current_fill[:3]  # Only take the first three elements
                self.set_fill_color(int(r), int(g), int(b))
            else:
                self.set_fill_color(255, 255, 255)  # Default white
            
            # Handle current_draw
            if isinstance(current_draw, tuple) and len(current_draw) >= 3:
                r, g, b = current_draw[:3]  # Only take the first three elements
                self.set_draw_color(int(r), int(g), int(b))
            else:
                self.set_draw_color(0, 0, 0)  # Default black
            
            # Handle current_text
            if isinstance(current_text, tuple) and len(current_text) >= 3:
                r, g, b = current_text[:3]  # Only take the first three elements
                self.set_text_color(int(r), int(g), int(b))
            else:
                self.set_text_color(0, 0, 0)  # Default black
        except Exception:
            # If any error occurs, use defaults
            self.set_fill_color(255, 255, 255)  # White
            self.set_draw_color(0, 0, 0)  # Black
            self.set_text_color(0, 0, 0)  # Black
        
        self.set_xy(current_x, new_y + 5)
        
    def insight_callout(self, title, text, icon=">>"):
        """Add an insight callout with title and description"""
        # Convert texts to latin-1 safe
        safe_title = self._encode_text(title)
        safe_text = self._encode_text(text)
        safe_icon = self._encode_text(icon)
        
        # Save current position and colors
        current_x = self.x
        current_y = self.y
        
        # Add the title with ASCII-compatible icon (no emoji)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        self.cell(0, 6, f"{safe_icon} {safe_title}", 0, 1)
        
        # Add the insight text
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(60, 60, 60)
        self.set_x(current_x + 5)
        self.multi_cell(0, 5, safe_text)
        self.ln(3)
    
    def narrative_paragraph(self, text):
        """Add a narrative paragraph with more space and styling"""
        # Convert to latin-1 safe text
        safe_text = self._encode_text(text)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, safe_text)
        self.ln(4)
    
    def metric_row(self, label, value, width1=80, width2=30, color=None):
        """Add a metric row with label and value"""
        # Convert label to latin-1 safe text
        safe_label = self._encode_text(label)
        
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(width1, 6, safe_label, 0, 0)
        
        # Safely convert value to string
        if value is None:
            value_str = "N/A"
        else:
            value_str = str(value)
            
        # Encode value string to latin-1
        safe_value_str = self._encode_text(value_str)
        
        # Apply color if provided
        if color:
            try:
                # Ensure color values are valid integers
                # Convert values safely to integers, handling edge cases
                r = int(color[0]) if len(color) > 0 and str(color[0]).strip() and not str(color[0]).isspace() else 0
                g = int(color[1]) if len(color) > 1 and str(color[1]).strip() and not str(color[1]).isspace() else 0
                b = int(color[2]) if len(color) > 2 and str(color[2]).strip() and not str(color[2]).isspace() else 0
                self.set_text_color(r, g, b)
            except (ValueError, TypeError, IndexError):
                # Fall back to default color if conversion fails
                self.set_text_color(0, 0, 0)
        
        self.cell(width2, 6, safe_value_str, 0, 1)
        self.set_text_color(0, 0, 0)
    
    def add_score_badge(self, score, x, y, size=20):
        """Add a colored score badge"""
        # Handle invalid scores
        try:
            # Handle various invalid score formats
            if score is None:
                score = 0.0
            elif isinstance(score, str):
                # Handle empty strings, whitespace, and percentage strings
                score_str = score.strip()
                if score_str == "":
                    score = 0.0
                elif score_str.endswith('%'):
                    # Remove % sign and convert to float
                    try:
                        score = float(score_str.rstrip('%'))
                    except (ValueError, TypeError):
                        score = 0.0
                else:
                    # Try to convert directly to float
                    score = float(score_str)
            else:
                score = float(score)
        except (ValueError, TypeError):
            # Default to zero if conversion fails
            score = 0.0
        
        # Ensure score is in valid range
        score = max(0.0, min(100.0, score))
        
        # Determine color based on score
        if score >= 90:
            color = (0, 128, 0)  # Green
        elif score >= 75:
            color = (255, 165, 0)  # Orange
        else:
            color = (200, 0, 0)  # Red
        
        # Save current position and settings
        current_fill = self.fill_color
        current_draw = self.draw_color
        current_x = self.x
        current_y = self.y
        
        # Draw circle with explicit color values to avoid LSP errors
        try:
            # Check if color is a tuple with at least 3 elements
            if isinstance(color, tuple) and len(color) >= 3:
                r, g, b = color[:3]  # Only take the first three elements
            else:
                # Fallback if color is not valid
                r, g, b = 0, 0, 0  # Default black
                
            self.set_fill_color(int(r), int(g), int(b))
            self.set_draw_color(int(r), int(g), int(b))
        except Exception:
            # If any error occurs, use default black
            self.set_fill_color(0, 0, 0)
            self.set_draw_color(0, 0, 0)
        self.ellipse(x, y, size, size, 'F')
        
        # Add score text
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(255, 255, 255)
        
        # Format score as integer if it's a whole number, otherwise with one decimal place
        try:
            if score == float(int(score)):
                score_text = f"{int(score)}"
            else:
                score_text = f"{score:.1f}"
        except (ValueError, TypeError):
            score_text = "0"
            
        # Encode score text to latin-1
        safe_score_text = self._encode_text(score_text)
        
        # Adjust position to center text in circle
        text_width = self.get_string_width(safe_score_text)
        self.set_xy(x + size/2 - text_width/2, y + size/2 - 4)
        self.cell(text_width, 8, safe_score_text, 0, 0, 'C')
        
        # Restore original position and settings
        self.set_xy(current_x, current_y)
        
        # Safely cast to int to avoid LSP errors, handling empty strings and whitespace
        fill_r = int(current_fill[0]) if isinstance(current_fill, tuple) and len(current_fill) > 0 and str(current_fill[0]).strip() and not str(current_fill[0]).isspace() else 0
        fill_g = int(current_fill[1]) if isinstance(current_fill, tuple) and len(current_fill) > 1 and str(current_fill[1]).strip() and not str(current_fill[1]).isspace() else 0
        fill_b = int(current_fill[2]) if isinstance(current_fill, tuple) and len(current_fill) > 2 and str(current_fill[2]).strip() and not str(current_fill[2]).isspace() else 0
        
        draw_r = int(current_draw[0]) if isinstance(current_draw, tuple) and len(current_draw) > 0 and str(current_draw[0]).strip() and not str(current_draw[0]).isspace() else 0
        draw_g = int(current_draw[1]) if isinstance(current_draw, tuple) and len(current_draw) > 1 and str(current_draw[1]).strip() and not str(current_draw[1]).isspace() else 0
        draw_b = int(current_draw[2]) if isinstance(current_draw, tuple) and len(current_draw) > 2 and str(current_draw[2]).strip() and not str(current_draw[2]).isspace() else 0
        
        self.set_fill_color(fill_r, fill_g, fill_b)
        self.set_draw_color(draw_r, draw_g, draw_b)
        self.set_text_color(0, 0, 0)
    
    def add_plot_from_json(self, plot_json):
        """Add a plotly plot from JSON string"""
        try:
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_filename = tmp_file.name
                
                # Convert plot JSON to figure and save as png
                fig = pio.from_json(plot_json)
                fig.write_image(temp_filename, width=800, height=500, scale=2)
                
                # Add image to PDF
                self.image(temp_filename, x=10, y=None, w=190)
                
                # Remove temp file
                os.remove(temp_filename)
                
                # Add space after image
                self.ln(5)
                
        except Exception as e:
            self.body_text(f"Error rendering visualization: {str(e)}")
    
    def add_cover_page(self):
        """Add a cover page to the report"""
        self.add_page()
        
        # Title
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        self.ln(30)
        safe_title = self._encode_text(self.title)
        self.multi_cell(0, 12, safe_title, 0, 'C')
        
        # Add organization if provided
        if self.organization:
            self.ln(10)
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(150, 70, 20)  # Darker orange
            safe_organization = self._encode_text(self.organization)
            self.cell(0, 10, safe_organization, 0, 1, 'C')
        
        # Add date
        self.ln(15)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(80, 80, 80)
        date_str = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        safe_date_str = self._encode_text(date_str)
        self.cell(0, 10, safe_date_str, 0, 1, 'C')
        
        # Add data quality icon/image as text
        self.ln(20)
        self.set_font('Helvetica', 'B', 40)
        self.set_text_color(238, 114, 3)  # Arcadis orange
        quality_text = self._encode_text("QUALITY")
        self.cell(0, 20, quality_text, 0, 1, 'C')
        
        # Add DAMA reference
        self.ln(30)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        dama_text = self._encode_text("Based on DAMA-DMBOK Data Quality Management Framework")
        self.cell(0, 5, dama_text, 0, 1, 'C')
    
    def add_summary_page(self, summary):
        """Add executive summary page with narrative storytelling"""
        self.add_page()
        self.chapter_title("Executive Summary")
        
        # Introduction narrative
        self.narrative_paragraph(
            "This executive summary provides an overview of the data quality assessment "
            "results, highlighting key findings, issues, and recommendations. The assessment "
            "follows the DAMA principles for comprehensive data quality evaluation across "
            "multiple dimensions including completeness, consistency, accuracy, uniqueness, "
            "timeliness, and validity."
        )
        
        # Overall quality score
        if "overall" in summary["overall_scores"]:
            overall_score = summary["overall_scores"]["overall"]
            
            # Determine color and narrative based on score
            if overall_score >= 90:
                color_text = "Excellent"
                color = (0, 128, 0)  # Green
                quality_narrative = (
                    f"The dataset demonstrates excellent overall quality with a score of {overall_score:.1f}/100. "
                    f"This indicates high reliability across most quality dimensions with only minor issues detected. "
                    f"The data is suitable for critical business operations and analytical purposes with minimal concerns."
                )
            elif overall_score >= 75:
                color_text = "Good"
                color = (255, 165, 0)  # Orange
                quality_narrative = (
                    f"The dataset shows good overall quality with a score of {overall_score:.1f}/100. "
                    f"While generally reliable, there are some quality issues that should be addressed. "
                    f"The data can be used for most business purposes but may require additional validation for critical applications."
                )
            else:
                color_text = "Needs Improvement"
                color = (200, 0, 0)  # Red
                quality_narrative = (
                    f"The dataset requires significant improvement with an overall quality score of {overall_score:.1f}/100. "
                    f"Multiple quality issues have been identified that could impact business decisions. "
                    f"Addressing the top issues identified in this report should be prioritized before using this data for critical purposes."
                )
            
            # Create a visual score display
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(238, 114, 3)  # Arcadis orange
            self.cell(120, 10, "Overall Data Quality Score:", 0, 0)
            
            # Format score display with proper error handling
            try:
                if overall_score is None:
                    score_display = "N/A"
                elif isinstance(overall_score, str):
                    # Try to convert string to float
                    try:
                        overall_score = float(overall_score.strip().rstrip('%'))
                        score_display = f"{overall_score:.1f}/100"
                    except (ValueError, TypeError):
                        score_display = "N/A"
                else:
                    # Try to convert to float and format
                    try:
                        overall_score = float(overall_score)
                        score_display = f"{overall_score:.1f}/100"
                    except (ValueError, TypeError):
                        score_display = "N/A"
            except Exception:
                score_display = "N/A"
            
            # Use explicit int casting for color components with error handling
            try:
                # Check if color is a tuple with at least 3 elements
                if isinstance(color, tuple) and len(color) >= 3:
                    r, g, b = color[:3]  # Only take the first three elements
                else:
                    # Fallback to black if color is not valid
                    r, g, b = 0, 0, 0
                    
                self.set_text_color(int(r), int(g), int(b))
            except Exception:
                # If any error occurs, use default black
                self.set_text_color(0, 0, 0)
                
            self.cell(30, 10, score_display, 0, 1)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, f"({color_text})", 0, 1, 'R')
            
            self.ln(5)
            
            # Add narrative description of overall quality
            self.narrative_paragraph(quality_narrative)
            
            # Add highlight box for quick takeaway with proper formatting
            try:
                # Format the score for highlight box
                if isinstance(overall_score, (int, float)):
                    score_text = f"{overall_score:.1f}/100"
                else:
                    score_text = "N/A"
                
                self.highlight_box(f"Overall Assessment: {color_text} - Score {score_text}")
            except Exception:
                # Fallback if there's any error
                self.highlight_box(f"Overall Assessment: {color_text}")
            self.ln(10)
        
        # Dimension scores with narrative context
        self.section_title("Quality Dimension Scores")
        
        dimension_names = {
            "completeness": "Completeness",
            "consistency": "Consistency",
            "accuracy": "Accuracy",
            "uniqueness": "Uniqueness",
            "timeliness": "Timeliness",
            "validity": "Validity"
        }
        
        dimension_descriptions = {
            "completeness": "measures the presence of required data in all fields",
            "consistency": "evaluates data conformity across the dataset",
            "accuracy": "assesses how well data reflects real-world values",
            "uniqueness": "identifies duplicate records and redundant data",
            "timeliness": "evaluates if data is up-to-date and relevant",
            "validity": "checks if data follows defined formats and rules"
        }
        
        # Add brief introduction to dimensions
        self.narrative_paragraph(
            "The following scores reflect performance across six key data quality dimensions. "
            "These dimensions are based on the DAMA framework and collectively provide a comprehensive view of data quality."
        )
        
        # Identify strongest and weakest dimensions
        strongest_dim = None
        weakest_dim = None
        strongest_score = -1
        weakest_score = 101
        
        for dimension in dimension_names:
            if dimension in summary["overall_scores"]:
                try:
                    # Safely get and convert the score using the same logic as above
                    raw_score = summary["overall_scores"][dimension]
                    
                    # Handle various types of values
                    if raw_score is None:
                        score = 0.0
                    elif isinstance(raw_score, str):
                        score_str = raw_score.strip()
                        if not score_str or score_str.isspace():
                            score = 0.0
                        elif score_str.endswith('%'):
                            try:
                                score = float(score_str.rstrip('%'))
                            except ValueError:
                                score = 0.0
                        else:
                            try:
                                score = float(score_str)
                            except ValueError:
                                score = 0.0
                    else:
                        score = float(raw_score)
                    
                    # Ensure score is in valid range
                    score = max(0.0, min(100.0, score))
                    
                    # Update strongest/weakest dimensions
                    if score > strongest_score:
                        strongest_score = score
                        strongest_dim = dimension
                    if score < weakest_score:
                        weakest_score = score
                        weakest_dim = dimension
                except Exception:
                    # Skip this dimension if there's an error
                    continue
        
        # Create a score table with narrative descriptions
        for dimension, score_key in dimension_names.items():
            if dimension in summary["overall_scores"]:
                # Safely get the score and convert to float
                try:
                    raw_score = summary["overall_scores"][dimension]
                    
                    # Handle various types of values
                    if raw_score is None:
                        score = 0.0
                    elif isinstance(raw_score, str):
                        # Handle percentage strings and empty strings
                        score_str = raw_score.strip()
                        if not score_str or score_str.isspace():
                            score = 0.0
                        elif score_str.endswith('%'):
                            # Remove % sign and convert
                            try:
                                score = float(score_str.rstrip('%'))
                            except ValueError:
                                score = 0.0
                        else:
                            # Try direct conversion
                            try:
                                score = float(score_str)
                            except ValueError:
                                score = 0.0
                    else:
                        # Try to convert to float directly
                        score = float(raw_score)
                
                    # Ensure score is in valid range
                    score = max(0.0, min(100.0, score))
                    
                    # Determine color based on score
                    if score >= 90:
                        color = (0, 128, 0)  # Green
                    elif score >= 75:
                        color = (255, 165, 0)  # Orange
                    else:
                        color = (200, 0, 0)  # Red
                    
                    # Format score display string with proper rounding
                    if score == float(int(score)):
                        score_display = f"{int(score)}/100"
                    else:
                        score_display = f"{score:.1f}/100"
                    
                    self.metric_row(f"{score_key}:", score_display, color=color)
                except Exception as e:
                    # Fallback for any conversion errors
                    self.metric_row(f"{score_key}:", "N/A", color=(100, 100, 100))
                
                # Special callout for strongest or weakest dimension
                try:
                    # Only display insights if we successfully parsed the score above
                    current_score = score  # Use the score from the try block above
                    if dimension == strongest_dim and strongest_score >= 75:
                        self.insight_callout(
                            f"Strength: {score_key}",
                            f"This dimension scored highest at {strongest_score:.1f}/100, indicating that the data {dimension_descriptions[dimension]} effectively."
                        )
                    elif dimension == weakest_dim and weakest_score < 75:
                        self.insight_callout(
                            f"Area for Improvement: {score_key}",
                            f"This dimension requires attention with a score of {weakest_score:.1f}/100. Improvements in how the data {dimension_descriptions[dimension]} should be prioritized."
                        )
                except Exception:
                    # Skip callouts if there's an error
                    pass
        
        self.ln(10)
        
        # Narrative summary of dimension scores
        if strongest_dim and weakest_dim:
            self.narrative_paragraph(
                f"The assessment shows that {dimension_names[strongest_dim].lower()} is the strongest aspect of this dataset, "
                f"while {dimension_names[weakest_dim].lower()} presents the most significant opportunity for improvement. "
                f"Addressing issues in the {dimension_names[weakest_dim].lower()} dimension would yield the greatest overall quality improvement."
            )
        
        # Top issues with narrative context
        if summary["top_issues"]:
            self.section_title("Top Data Quality Issues")
            
            # Introduction to issues
            self.narrative_paragraph(
                "The following issues were identified as the most significant quality concerns. "
                "These issues have been prioritized based on their impact on data usability and business operations."
            )
            
            for i, issue in enumerate(summary["top_issues"][:5], 1):
                sheet = issue["sheet"]
                column = issue["column"]
                issue_desc = issue["issue"]
                dimension = issue["dimension"]
                
                # Use more formatted and descriptive presentation
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(150, 70, 20)  # Darker orange
                self.cell(0, 6, f"Issue #{i}: {sheet} - {column}", 0, 1)
                
                self.set_font('Helvetica', '', 10)
                self.set_text_color(0, 0, 0)
                self.cell(10, 5, "", 0, 0)
                self.multi_cell(0, 5, f"{issue_desc} ({dimension})")
                
                # Add contextual information for the first major issue
                if i == 1:
                    self.insight_callout(
                        "Impact Analysis",
                        f"This {dimension} issue in {sheet}.{column} may affect business operations that rely on this data. "
                        f"Addressing this should be considered a priority for data quality improvement."
                    )
                
                self.ln(2)
            
            # Narrative summary of issues
            num_issues = len(summary["top_issues"][:5])
            self.narrative_paragraph(
                f"The assessment identified {num_issues} critical data quality issues that require attention. "
                f"These issues are concentrated in the {summary['top_issues'][0]['dimension']} dimension, "
                f"suggesting a systematic improvement opportunity in this area."
            )
            
            self.ln(5)
        
        # Recommendations with narrative context
        if summary["recommendations"]:
            self.section_title("Key Recommendations")
            
            # Introduction to recommendations
            self.narrative_paragraph(
                "Based on the assessment results, the following recommendations are provided to improve data quality. "
                "These recommendations are prioritized based on their expected impact and implementation feasibility."
            )
            
            for i, rec in enumerate(summary["recommendations"][:5], 1):
                dimension = rec["dimension"]
                recommendation = rec["recommendation"]
                priority = rec["priority"]
                
                # Priority color
                if priority == "High":
                    priority_color = (200, 0, 0)  # Red
                elif priority == "Medium":
                    priority_color = (255, 165, 0)  # Orange
                else:
                    priority_color = (0, 128, 0)  # Green
                
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(150, 70, 20)  # Darker orange
                self.cell(0, 6, f"Recommendation #{i}: {dimension}", 0, 1)
                
                self.set_font('Helvetica', '', 10)
                self.set_text_color(0, 0, 0)
                self.cell(10, 5, "", 0, 0)
                self.multi_cell(0, 5, recommendation)
                
                # Use explicit int casting for priority color with error handling
                try:
                    # Check if priority_color is a tuple with at least 3 elements
                    if isinstance(priority_color, tuple) and len(priority_color) >= 3:
                        r, g, b = priority_color[:3]  # Only take the first three elements
                    else:
                        # Fallback to black if color is not valid
                        r, g, b = 0, 0, 0
                        
                    self.set_font('Helvetica', 'B', 9)
                    self.set_text_color(int(r), int(g), int(b))
                except Exception:
                    # If any error occurs, use default black
                    self.set_font('Helvetica', 'B', 9)
                    self.set_text_color(0, 0, 0)
                
                self.cell(0, 5, f"Priority: {priority}", 0, 1, 'R')
                
                # Add implementation guidance for high priority recommendations
                if priority == "High":
                    self.insight_callout(
                        "Implementation Note",
                        f"Implementing this {dimension} recommendation would address multiple quality issues and should be considered for immediate action."
                    )
                
                self.ln(4)
            
            # Summary of recommendations
            high_priority = sum(1 for rec in summary["recommendations"][:5] if rec["priority"] == "High")
            self.narrative_paragraph(
                f"The assessment provides {len(summary['recommendations'][:5])} actionable recommendations, "
                f"with {high_priority} identified as high priority. Implementing these recommendations in order of priority "
                f"will systematically improve the overall data quality score."
            )
        
        # Add DAMA context
        self.ln(10)
        self.section_title("About DAMA Framework")
        
        self.narrative_paragraph(
            "This assessment follows the DAMA-DMBOK (Data Management Body of Knowledge) framework, "
            "which provides industry-standard approaches for evaluating and improving data quality. "
            "The multi-dimensional approach ensures that data is evaluated from various perspectives, "
            "providing a holistic view of its fitness for purpose."
        )
        
        self.info_text(
            "DAMA International's data quality dimensions include completeness, consistency, accuracy, "
            "validity, timeliness, and uniqueness, which collectively assess whether data is fit for its "
            "intended uses in operations, planning, and decision-making."
        )

def generate_pdf_report(assessment_results, file_info, report_title, organization="", progress_callback=None):
    """
    Generate a PDF report from assessment results
    
    Args:
        assessment_results: Dictionary containing assessment results
        file_info: Dictionary containing file information
        report_title: Title for the report
        organization: Organization name
        progress_callback: Callback function for progress updates
    
    Returns:
        Path to the generated PDF file
    """
    if progress_callback:
        progress_callback(0, "Initializing PDF report...")
    
    # Create PDF report
    pdf = DataQualityReport(title=report_title, organization=organization, orientation='P', unit='mm', format='A4')
    
    # No need to add custom fonts, using standard Helvetica
    
    # Add cover page
    pdf.add_cover_page()
    
    if progress_callback:
        progress_callback(10, "Adding executive summary...")
    
    # Add summary page if available
    if "summary" in assessment_results:
        pdf.add_summary_page(assessment_results["summary"])
    
    # Add file information page
    pdf.add_page()
    pdf.chapter_title("Data Overview")
    
    # Add file information
    pdf.section_title("File Information")
    pdf.metric_row("File Type:", file_info["file_type"])
    pdf.metric_row("Number of Sheets:", str(len(file_info["sheet_names"])))
    pdf.metric_row("Total Rows (all sheets):", str(file_info["total_rows"]))
    pdf.metric_row("Total Columns (all sheets):", str(file_info["total_columns"]))
    
    # Add sheet details
    pdf.section_title("Sheet Details")
    for sheet, details in file_info["sheets"].items():
        pdf.sub_section_title(sheet)
        pdf.metric_row("Rows:", str(details["rows"]), width1=60, width2=30)
        pdf.metric_row("Columns:", str(details["columns"]), width1=60, width2=30)
        pdf.metric_row("Memory Usage:", f"{details['memory_usage']:.2f} MB", width1=60, width2=30)
    
    # Process each sheet
    sheet_count = len(assessment_results) - (1 if "summary" in assessment_results else 0)
    sheet_idx = 0
    
    for sheet_name, sheet_results in assessment_results.items():
        if sheet_name == "summary":
            continue
        
        sheet_idx += 1
        if progress_callback:
            progress_callback(10 + (sheet_idx / sheet_count) * 80, f"Processing sheet: {sheet_name}...")
        
        # Add sheet assessment page
        pdf.add_page()
        pdf.chapter_title(f"Assessment: {sheet_name}")
        
        # Add overview section with overall scores
        pdf.section_title("Quality Overview")
        
        overall_scores = {}
        
        if "completeness" in sheet_results:
            overall_scores["Completeness"] = sheet_results["completeness"]["overall"]["completeness_score"]
        
        if "consistency" in sheet_results:
            overall_scores["Consistency"] = sheet_results["consistency"]["overall"]["overall_consistency_score"]
        
        if "accuracy" in sheet_results:
            overall_scores["Accuracy"] = sheet_results["accuracy"]["overall"]["overall_accuracy_score"]
        
        if "uniqueness" in sheet_results:
            overall_scores["Uniqueness"] = sheet_results["uniqueness"]["overall"]["uniqueness_score"]
        
        if "timeliness" in sheet_results and sheet_results["timeliness"]["overall"]["timeliness_score"] is not None:
            overall_scores["Timeliness"] = sheet_results["timeliness"]["overall"]["timeliness_score"]
        
        if "validity" in sheet_results:
            overall_scores["Validity"] = sheet_results["validity"]["overall"]["overall_validity_score"]
        
        # Create a score table with colored badges
        y_pos = pdf.y
        for i, (dimension, raw_score) in enumerate(overall_scores.items()):
            try:
                # Handle various types of values
                if raw_score is None:
                    score = 0.0
                    score_display = "N/A"
                elif isinstance(raw_score, str):
                    # Handle percentage strings and empty strings
                    score_str = raw_score.strip()
                    if not score_str or score_str.isspace():
                        score = 0.0
                        score_display = "N/A"
                    elif score_str.endswith('%'):
                        # Remove % sign and convert
                        try:
                            score = float(score_str.rstrip('%'))
                            score_display = f"{score:.1f}/100"
                        except ValueError:
                            score = 0.0
                            score_display = "N/A"
                    else:
                        # Try direct conversion
                        try:
                            score = float(score_str)
                            score_display = f"{score:.1f}/100"
                        except ValueError:
                            score = 0.0
                            score_display = "N/A"
                else:
                    # Try to convert to float directly
                    score = float(raw_score)
                    score_display = f"{score:.1f}/100"
                
                # Ensure score is in valid range for badge
                score = max(0.0, min(100.0, score))
                
                # Display row and badge
                pdf.metric_row(f"{dimension}:", score_display)
                pdf.add_score_badge(score, 160, y_pos - 1)
            except Exception:
                # Fallback for any errors
                pdf.metric_row(f"{dimension}:", "N/A")
            
            y_pos += 6
        
        # Add radar chart visualization if available
        if "visualizations" in sheet_results and "quality_dimensions_radar" in sheet_results["visualizations"]:
            pdf.ln(5)
            pdf.add_plot_from_json(sheet_results["visualizations"]["quality_dimensions_radar"])
        
        # Add completeness section
        if "completeness" in sheet_results:
            pdf.add_page()
            pdf.chapter_title(f"Completeness: {sheet_name}")
            
            completeness = sheet_results["completeness"]
            overall_completeness = completeness["overall"]["completeness_percentage"]
            
            pdf.body_text(
                f"Overall completeness of the data is {overall_completeness:.2f}%. "
                f"There are {completeness['overall']['missing_cells']} missing values out of "
                f"{completeness['overall']['total_cells']} total cells."
            )
            
            # Show top incomplete columns
            if completeness["top_incomplete"]:
                pdf.section_title("Columns with Missing Values")
                
                for col, pct in completeness["top_incomplete"].items():
                    missing_pct = 100 - pct
                    pdf.metric_row(f"{col}:", f"{missing_pct:.2f}% missing")
            
            # Add missing values visualization if available
            if "visualizations" in sheet_results and "missing_values_plot" in sheet_results["visualizations"]:
                pdf.ln(5)
                pdf.add_plot_from_json(sheet_results["visualizations"]["missing_values_plot"])
        
        # Add data type distribution visualization if available
        if "visualizations" in sheet_results and "datatype_distribution" in sheet_results["visualizations"]:
            pdf.add_page()
            pdf.chapter_title(f"Data Structure: {sheet_name}")
            pdf.section_title("Data Type Distribution")
            pdf.add_plot_from_json(sheet_results["visualizations"]["datatype_distribution"])
        
        # Add consistency section
        if "consistency" in sheet_results:
            pdf.add_page()
            pdf.chapter_title(f"Consistency: {sheet_name}")
            
            consistency = sheet_results["consistency"]
            
            pdf.body_text(
                f"Data type consistency score: {consistency['overall']['type_consistency_score']:.2f}/100\n"
                f"Value range consistency score: {consistency['overall']['value_consistency_score']:.2f}/100"
            )
            
            # Show type consistency issues
            mixed_types = {col: data for col, data in consistency["type_consistency"].items() 
                          if data.get("mixed_types", False)}
            
            if mixed_types:
                pdf.section_title("Columns with Mixed Data Types")
                
                for col, data in mixed_types.items():
                    pdf.sub_section_title(col)
                    pdf.info_text(f"Types found: {', '.join(data['types_found'])}")
            
            # Show value consistency issues
            if "value_consistency" in consistency:
                outlier_cols = {col: data for col, data in consistency["value_consistency"].items() 
                              if data.get("outlier_percentage", 0) > 1}
                
                if outlier_cols:
                    pdf.section_title("Columns with Value Range Issues")
                    
                    for col, data in outlier_cols.items():
                        pdf.sub_section_title(col)
                        pdf.info_text(
                            f"Range: {data['min']} to {data['max']}\n"
                            f"Outliers: {data['outlier_count']} ({data['outlier_percentage']:.2f}%)"
                        )
        
        # Add numeric distributions if available
        if "visualizations" in sheet_results and "numeric_distributions" in sheet_results["visualizations"]:
            pdf.add_page()
            pdf.chapter_title(f"Value Distributions: {sheet_name}")
            pdf.add_plot_from_json(sheet_results["visualizations"]["numeric_distributions"])
        
        # Add top issues chart if available
        if "visualizations" in sheet_results and "top_issues_chart" in sheet_results["visualizations"]:
            pdf.add_page()
            pdf.chapter_title(f"Quality Issues: {sheet_name}")
            pdf.add_plot_from_json(sheet_results["visualizations"]["top_issues_chart"])
        
        # Add uniqueness section
        if "uniqueness" in sheet_results:
            pdf.add_page()
            pdf.chapter_title(f"Uniqueness: {sheet_name}")
            
            uniqueness = sheet_results["uniqueness"]
            
            pdf.body_text(
                f"Overall uniqueness score: {uniqueness['overall']['uniqueness_score']:.2f}/100\n"
                f"There are {uniqueness['overall']['duplicate_rows']} duplicate rows out of "
                f"{uniqueness['overall']['total_rows']} total rows."
            )
            
            # Show potential key columns
            if uniqueness["key_candidates"]:
                pdf.section_title("Potential Key Columns")
                pdf.body_text(
                    "The following columns have unique values for each row and "
                    "could serve as primary keys:"
                )
                
                for col in uniqueness["key_candidates"]:
                    pdf.info_text(f"* {col}")
            
            # Show high cardinality non-key columns
            if uniqueness["high_cardinality_non_keys"]:
                pdf.section_title("High Cardinality Non-Key Columns")
                pdf.body_text(
                    "These columns have high cardinality but are not unique. "
                    "They may have data quality issues:"
                )
                
                for item in uniqueness["high_cardinality_non_keys"]:
                    pdf.info_text(
                        f"* {item['column']}: {item['uniqueness_ratio']*100:.2f}% unique "
                        f"({item['unique_values']} values)"
                    )
        
        # Add timeliness section
        if "timeliness" in sheet_results and sheet_results["timeliness"]["overall"]["timeliness_score"] is not None:
            pdf.add_page()
            pdf.chapter_title(f"Timeliness: {sheet_name}")
            
            timeliness = sheet_results["timeliness"]
            
            pdf.body_text(
                f"Overall timeliness score: {timeliness['overall']['timeliness_score']:.2f}/100\n"
                f"Number of date columns analyzed: {timeliness['overall']['date_columns_found']}"
            )
            
            # Show details for each date column
            if "column_timeliness" in timeliness:
                pdf.section_title("Date Column Analysis")
                
                for col, data in timeliness["column_timeliness"].items():
                    pdf.sub_section_title(col)
                    
                    if data["min_date"] and data["max_date"]:
                        pdf.info_text(
                            f"Date range: {data['min_date']} to {data['max_date']}\n"
                            f"Time span: {data['time_span_days']} days\n"
                            f"Recency: {data['recency_days']} days old\n"
                            f"Timeliness category: {data['timeliness_category']}"
                        )
                    else:
                        pdf.info_text("No valid dates found in this column.")
        
        # Add validity section
        if "validity" in sheet_results:
            pdf.add_page()
            pdf.chapter_title(f"Validity: {sheet_name}")
            
            validity = sheet_results["validity"]
            
            pdf.body_text(
                f"Overall validity score: {validity['overall']['overall_validity_score']:.2f}/100\n"
                f"Number of checks performed: {validity['overall']['checks_performed']}"
            )
            
            # Show validity check results
            if "validity_checks" in validity and validity["validity_checks"]:
                pdf.section_title("Format Validation Checks")
                
                for col, data in validity["validity_checks"].items():
                    check_type = data["check_type"].replace("_", " ").title()
                    validity_pct = data["validity_percentage"]
                    
                    pdf.sub_section_title(f"{col} ({check_type})")
                    
                    # Determine status text and color
                    if validity_pct >= 95:
                        status = "Valid"
                        color = (0, 128, 0)  # Green
                    elif validity_pct >= 80:
                        status = "Mostly Valid"
                        color = (255, 165, 0)  # Orange
                    else:
                        status = "Invalid"
                        color = (200, 0, 0)  # Red
                    
                    pdf.metric_row("Validity:", f"{validity_pct:.2f}%", color=color)
                    pdf.metric_row("Status:", status, color=color)
    
    # Add DAMA framework explanation
    if progress_callback:
        progress_callback(90, "Adding DAMA framework explanation...")
    
    pdf.add_page()
    pdf.chapter_title("DAMA Data Quality Framework")
    
    pdf.body_text(
        "The DAMA-DMBOK (Data Management Body of Knowledge) framework defines data quality as "
        "the planning, implementation, and control of activities that apply quality management "
        "techniques to data, in order to assure it is fit for consumption and meets the needs of "
        "data consumers."
    )
    
    pdf.section_title("Data Quality Dimensions")
    
    # Completeness
    pdf.sub_section_title("Completeness")
    pdf.body_text(
        "The degree to which all required data is present. Completeness measures whether all "
        "expected attributes are provided, with no missing or null values."
    )
    
    # Consistency
    pdf.sub_section_title("Consistency")
    pdf.body_text(
        "The degree to which data values are consistent across data sets and systems. "
        "Consistency ensures data follows the same format and structure without contradictions "
        "or variations."
    )
    
    # Accuracy
    pdf.sub_section_title("Accuracy")
    pdf.body_text(
        "The degree to which data correctly reflects the real-world objects or events being described. "
        "Accuracy measures how well data represents the true value of the intended attribute."
    )
    
    # Uniqueness
    pdf.sub_section_title("Uniqueness")
    pdf.body_text(
        "The degree to which data values are unique when they should be, with no unintended "
        "duplications. Uniqueness ensures entities are represented only once within the dataset."
    )
    
    # Timeliness
    pdf.sub_section_title("Timeliness")
    pdf.body_text(
        "The degree to which data is available when required. Timeliness measures how current "
        "the data is and whether it's available within an acceptable timeframe."
    )
    
    # Validity
    pdf.sub_section_title("Validity")
    pdf.body_text(
        "The degree to which data conforms to defined business rules and format requirements. "
        "Validity ensures data follows specified syntax, type, range, and other constraints."
    )
    
    # Save the PDF to a temporary file
    if progress_callback:
        progress_callback(95, "Finalizing PDF report...")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_file.name
    temp_file.close()
    
    pdf.output(pdf_path)
    
    if progress_callback:
        progress_callback(100, "PDF report successfully generated!")
    
    return pdf_path
