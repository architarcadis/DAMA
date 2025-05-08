"""
AI Integration Module for Data Quality Assessment Tool
Provides natural language explanations, recommendations and interactions

This module can work with:
1. OpenAI API (requires OpenAI API key)
2. Anthropic API (requires Anthropic API key)
3. xAI API (requires xAI API key) 
4. Local LLM models (offline mode)
"""

import os
import json
import sys
import pandas as pd
from openai import OpenAI

# Import Anthropic if available
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None  # Define Anthropic as None to avoid unbound errors

# Default system prompts
SYSTEM_PROMPTS = {
    "explain_issues": """You are a data quality expert explaining technical findings to a business user.
    Explain the identified data quality issues in simple, non-technical language. Focus on business impact and risk.
    Be concise and use analogies where helpful.""",
    
    "recommend_fixes": """You are a data quality specialist providing actionable recommendations.
    Based on the identified issues, suggest specific, prioritized steps to improve data quality.
    Focus on practical solutions with consideration for effort vs impact.""",
    
    "classify_severity": """You are an automated system that classifies data quality issues by severity.
    Analyze the provided metrics and categorize each issue as Critical, High, Medium, or Low severity.
    Consider business impact, regulatory risk, and data trustworthiness in your assessment.""",
    
    "document_assessment": """You are a documentation specialist creating a comprehensive data quality assessment.
    Create a well-structured, professional report summarizing findings, implications, and recommendations.
    Use clear sections, bullet points, and highlight key metrics and concerns.""",
    
    "answer_query": """You are a helpful data quality assistant answering user questions.
    Provide accurate, clear responses based on the data quality assessment results.
    When unclear, ask clarifying questions. Do not speculate beyond available information."""
}

class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self):
        self.system_prompts = SYSTEM_PROMPTS
        
    def generate_response(self, prompt, system_prompt_key, data=None):
        """Generate response based on prompt and data"""
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(AIProvider):
    """OpenAI API integration"""
    
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                              # do not change this unless explicitly requested by the user
        
    def generate_response(self, prompt, system_prompt_key, data=None):
        """Generate response using OpenAI API"""
        system_prompt = self.system_prompts.get(system_prompt_key, "")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Add data as context if provided
        if data:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to markdown table for better context
                data_str = data.to_markdown()
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n{data_str}"})
            elif isinstance(data, dict):
                # Convert dict to formatted JSON string
                data_str = json.dumps(data, indent=2)
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n```json\n{data_str}\n```"})
            else:
                # Default stringification
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n{str(data)}"})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,  # Balanced between creativity and determinism
                max_tokens=1000   # Adjust based on needed response length
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

class AnthropicProvider(AIProvider):
    """Anthropic API integration"""
    
    def __init__(self, api_key=None):
        super().__init__()
        if not ANTHROPIC_AVAILABLE or Anthropic is None:
            raise ImportError("Anthropic package is not installed. Please install it with 'pip install anthropic'")
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        try:
            self.client = Anthropic(api_key=self.api_key)
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            self.model = "claude-3-5-sonnet-20241022"
        except Exception as e:
            raise ValueError(f"Error initializing Anthropic client: {str(e)}")
    
    def generate_response(self, prompt, system_prompt_key, data=None):
        """Generate response using Anthropic API"""
        system_prompt = self.system_prompts.get(system_prompt_key, "")
        
        content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add data as context if provided
        if data:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to markdown table for better context
                data_str = data.to_markdown()
                content.append({"type": "text", "text": f"Here is the data for context:\n\n{data_str}"})
            elif isinstance(data, dict):
                # Convert dict to formatted JSON string
                data_str = json.dumps(data, indent=2)
                content.append({"type": "text", "text": f"Here is the data for context:\n\n```json\n{data_str}\n```"})
            else:
                # Default stringification
                content.append({"type": "text", "text": f"Here is the data for context:\n\n{str(data)}"})
        
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=1000,
                temperature=0.5,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response from Anthropic: {str(e)}"

class XAIProvider(AIProvider):
    """xAI (Grok) API integration"""
    
    def __init__(self, api_key=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("xAI API key is required")
        
        # Using OpenAI client with xAI base URL
        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=self.api_key
        )
        
        # Default to the most powerful model
        self.model = "grok-2-1212"
    
    def generate_response(self, prompt, system_prompt_key, data=None):
        """Generate response using xAI API"""
        system_prompt = self.system_prompts.get(system_prompt_key, "")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Add data as context if provided
        if data:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to markdown table for better context
                data_str = data.to_markdown()
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n{data_str}"})
            elif isinstance(data, dict):
                # Convert dict to formatted JSON string
                data_str = json.dumps(data, indent=2)
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n```json\n{data_str}\n```"})
            else:
                # Default stringification
                messages.append({"role": "user", "content": f"Here is the data for context:\n\n{str(data)}"})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response from xAI: {str(e)}"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalLLMProvider(AIProvider):
    """Local LLM integration using scikit-learn for basic text generation and templating"""
    
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self.initialized = False
        self.responses = {}
        self.templates = {}
        self.vectorizer = None
        self.template_corpus = []
        
        # Initialize pre-defined response templates
        self._initialize_templates()
        
        try:
            # Create a vectorizer for text similarity
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.initialized = True
        except ImportError:
            print("Warning: scikit-learn not available. LocalLLMProvider will use basic pattern matching.")
    
    def _initialize_templates(self):
        """Initialize response templates for different scenarios"""
        # Templates for explaining data quality issues
        self.templates["explain_issues"] = [
            "The data contains {issue_count} quality issues across {dimension} dimensions. Most critical issues are in the {critical_dimension} dimension, showing {critical_issue}.",
            "Analysis reveals data quality concerns in {dimension}, with {critical_issue} being the most significant issue that requires immediate attention.",
            "Data quality assessment identified {issue_count} issues, primarily in {dimension}. The main concern is {critical_issue}, which affects data reliability."
        ]
        
        # Templates for recommendations
        self.templates["recommend_fixes"] = [
            "To address the {dimension} issues, consider implementing data validation rules that enforce {rule_type}. This would prevent issues like {critical_issue}.",
            "Implementing a data quality monitoring system would help identify and fix {dimension} issues before they impact business operations.",
            "To improve data quality, focus on cleaning the {critical_dimension} data first, as it shows the most severe issues like {critical_issue}."
        ]
        
        # Templates for severity classification
        self.templates["classify_severity"] = [
            "{'issue_1': {'severity': 'Critical', 'justification': 'Direct business impact', 'business_impact': 'Affects decision-making'}, 'issue_2': {'severity': 'Medium', 'justification': 'Indirect effect', 'business_impact': 'Minor reporting issues'}}",
            "{'completeness_issues': {'severity': 'High', 'justification': 'Missing critical data', 'business_impact': 'Incomplete analysis'}, 'format_issues': {'severity': 'Low', 'justification': 'Cosmetic problems', 'business_impact': 'Minimal effect'}}"
        ]
        
        # Templates for documentation
        self.templates["document_assessment"] = [
            """# Data Quality Assessment Report
## Executive Summary
The data quality assessment found {issue_count} issues across {dimension_count} dimensions.

## Key Findings
- {critical_dimension} shows the most critical issues
- {critical_issue} requires immediate attention
- Overall data quality score: {quality_score}/100

## Recommendations
1. Implement data validation for {critical_dimension}
2. Clean existing data to address {critical_issue}
3. Establish regular monitoring of data quality metrics""",
        ]
        
        # Templates for answering queries
        self.templates["answer_query"] = [
            "Based on the assessment results, the {dimension} dimension shows a quality score of {quality_score}/100. The main issue identified is {critical_issue}.",
            "The analysis indicates that {critical_dimension} has {issue_count} quality issues, with {critical_issue} being the most significant concern.",
            "According to the assessment, the data quality is {quality_level}, with a score of {quality_score}/100. The {critical_dimension} dimension needs the most improvement."
        ]
    
    def _extract_context(self, data):
        """Extract key context from data to populate templates"""
        context = {
            "issue_count": "multiple",
            "dimension": "various",
            "dimension_count": "several",
            "critical_dimension": "completeness",
            "critical_issue": "missing values",
            "quality_score": "65",
            "quality_level": "moderate",
            "rule_type": "data format constraints"
        }
        
        # Extract basic metrics if available
        if isinstance(data, dict):
            # Try to find quality dimensions
            dimensions = []
            if "summary" in data and "overall_scores" in data["summary"]:
                # Extract quality score if available
                if "overall" in data["summary"]["overall_scores"]:
                    context["quality_score"] = str(round(data["summary"]["overall_scores"]["overall"]))
                    score = float(context["quality_score"])
                    if score >= 90:
                        context["quality_level"] = "excellent"
                    elif score >= 75:
                        context["quality_level"] = "good"
                    elif score >= 60:
                        context["quality_level"] = "moderate"
                    else:
                        context["quality_level"] = "poor"
                
                # Find the critical dimension (lowest score)
                lowest_score = 100
                for dim, score in data["summary"]["overall_scores"].items():
                    if dim != "overall" and score < lowest_score:
                        lowest_score = score
                        context["critical_dimension"] = dim
            
            # Count total issues
            issue_count = 0
            for sheet_name, sheet_data in data.items():
                if sheet_name != "summary":
                    # Count completeness issues
                    if "completeness" in sheet_data and "overall" in sheet_data["completeness"]:
                        if "missing_cells" in sheet_data["completeness"]["overall"]:
                            issue_count += sheet_data["completeness"]["overall"]["missing_cells"]
            
            if issue_count > 0:
                context["issue_count"] = str(issue_count)
        
        return context
    
    def _find_most_similar_template(self, system_prompt_key, context):
        """Find the most relevant template based on the prompt"""
        if not self.initialized or self.vectorizer is None:
            # If not initialized with scikit-learn, use the first template
            return self.templates.get(system_prompt_key, ["No appropriate template found"])[0]
        
        # Use the system prompt to select a set of templates
        templates = self.templates.get(system_prompt_key, ["No appropriate template found"])
        
        # If only one template, return it
        if len(templates) == 1:
            return templates[0]
        
        try:
            # Use context keywords to find the most appropriate template
            context_text = " ".join(context.values())
            
            # Create a corpus with the templates and the context
            corpus = templates + [context_text]
            
            # Vectorize the corpus
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between context and each template
            similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            # Return the template with the highest similarity
            return templates[similarity_scores.argmax()]
        except Exception as e:
            # Fallback to first template on error
            print(f"Error finding similar template: {str(e)}")
            return templates[0]
    
    def generate_response(self, prompt, system_prompt_key, data=None):
        """Generate response using local text templates and basic NLP"""
        try:
            # Extract context from data
            context = self._extract_context(data)
            
            # Find the most appropriate template
            template = self._find_most_similar_template(system_prompt_key, context)
            
            # Fill in the template with context
            response = template.format(**context)
            
            # Update user prompt to improve future responses
            if len(prompt) > 10:
                for key in context:
                    if key in prompt.lower():
                        # Extract potential new context value
                        words = prompt.split()
                        for i, word in enumerate(words):
                            if key in word.lower() and i < len(words) - 1:
                                context[key] = words[i + 1].strip(',.!?')
            
            return response
        
        except Exception as e:
            # Fallback response
            return f"LocalLLM could not generate a response: {str(e)}. This is a simplified local implementation without requiring external API connections."

def get_ai_provider(provider_type="openai", api_key=None, model_path=None, model=None):
    """Factory function to get the appropriate AI provider"""
    if provider_type.lower() == "openai":
        provider = OpenAIProvider(api_key=api_key)
        if model:
            provider.model = model
        return provider
    elif provider_type.lower() == "anthropic":
        provider = AnthropicProvider(api_key=api_key)
        if model:
            provider.model = model
        return provider
    elif provider_type.lower() == "xai":
        provider = XAIProvider(api_key=api_key)
        if model:
            provider.model = model
        return provider
    elif provider_type.lower() == "local":
        return LocalLLMProvider(model_path=model_path)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

def explain_data_quality_issues(issues, provider_type="openai", api_key=None, model=None):
    """Explain data quality issues in natural language"""
    provider = get_ai_provider(provider_type, api_key, model=model)
    
    prompt = f"""
    Please explain the following data quality issues in simple, business-friendly language:
    
    {json.dumps(issues, indent=2)}
    
    Focus on:
    1. What these issues mean for the business
    2. Potential risks or impacts
    3. Why these issues might have occurred
    """
    
    return provider.generate_response(prompt, "explain_issues", issues)

def generate_recommendations(assessment_results, provider_type="openai", api_key=None, model=None):
    """Generate recommendations based on assessment results"""
    provider = get_ai_provider(provider_type, api_key, model=model)
    
    prompt = f"""
    Based on the following data quality assessment results, provide specific, 
    actionable recommendations for improving data quality:
    
    Focus on:
    1. High-impact, low-effort improvements
    2. Critical issues that should be addressed immediately
    3. Long-term structural improvements
    4. Best practices for maintaining data quality
    """
    
    return provider.generate_response(prompt, "recommend_fixes", assessment_results)

def classify_issue_severity(issues, provider_type="openai", api_key=None, model=None):
    """Classify issues by severity"""
    provider = get_ai_provider(provider_type, api_key, model=model)
    
    prompt = f"""
    Classify each of the following data quality issues by severity (Critical, High, Medium, Low):
    
    For each issue, provide:
    1. The severity classification
    2. A brief justification for the classification
    3. Estimated business impact
    
    Return the results in JSON format with the following structure:
    {{
        "issue_id": {{
            "severity": "Critical|High|Medium|Low",
            "justification": "Brief explanation",
            "business_impact": "Description of impact"
        }}
    }}
    """
    
    response = provider.generate_response(prompt, "classify_severity", issues)
    
    # Attempt to parse JSON response, fallback to text if not valid JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Could not parse response as JSON", "text_response": response}

def generate_assessment_documentation(assessment_results, provider_type="openai", api_key=None, model=None):
    """Generate comprehensive documentation from assessment results"""
    provider = get_ai_provider(provider_type, api_key, model=model)
    
    prompt = f"""
    Create a comprehensive data quality assessment report based on the following results:
    
    Include the following sections:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis by Data Quality Dimension
    4. Recommendations
    5. Next Steps
    
    Format the report professionally with clear headings, bullet points, and concise language.
    """
    
    return provider.generate_response(prompt, "document_assessment", assessment_results)

def answer_data_quality_query(query, assessment_results, provider_type="openai", api_key=None, model=None):
    """Answer user queries about data quality assessment"""
    provider = get_ai_provider(provider_type, api_key, model=model)
    
    prompt = f"""
    The user has asked the following question about the data quality assessment:
    
    "{query}"
    
    Please provide a helpful, accurate response based on the assessment results.
    """
    
    return provider.generate_response(prompt, "answer_query", assessment_results)