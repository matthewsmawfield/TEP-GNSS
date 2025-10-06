#!/usr/bin/env python3
"""
GCP Configuration Management for TEP-GNSS
=========================================

Handles secure loading of GCP credentials from environment variables
or .env files, with fallback to user prompts for missing values.

Author: Matthew Lukin Smawfield
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


class GCPConfig:
    """Secure GCP configuration management"""
    
    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize GCP configuration.
        
        Args:
            env_file: Path to .env file (defaults to .env.local, then .env)
        """
        self.project_root = Path(__file__).resolve().parents[2]
        
        # Try to load environment variables from .env files
        if env_file:
            load_dotenv(env_file)
        else:
            # Try .env.local first (for personal config), then .env
            local_env = self.project_root / '.env.local'
            default_env = self.project_root / '.env'
            
            if local_env.exists():
                load_dotenv(local_env)
            elif default_env.exists():
                load_dotenv(default_env)
    
    def get_project_id(self) -> str:
        """Get GCP project ID with fallback to user input"""
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            project_id = input("Enter your GCP Project ID: ").strip()
            if not project_id:
                raise ValueError("GCP Project ID is required")
        return project_id
    
    def get_zone(self) -> str:
        """Get GCP zone with fallback to user input"""
        zone = os.getenv('GCP_ZONE')
        if not zone:
            zone = input("Enter your GCP Zone (e.g., us-central1-a): ").strip()
            if not zone:
                raise ValueError("GCP Zone is required")
        return zone
    
    def get_instance_name(self) -> str:
        """Get GCP instance name with fallback to user input"""
        instance_name = os.getenv('GCP_INSTANCE_NAME')
        if not instance_name:
            instance_name = input("Enter your GCP Instance Name: ").strip()
            if not instance_name:
                raise ValueError("GCP Instance Name is required")
        return instance_name
    
    def get_all_config(self) -> Dict[str, str]:
        """Get all GCP configuration as a dictionary"""
        return {
            'GCP_PROJECT_ID': self.get_project_id(),
            'GCP_ZONE': self.get_zone(),
            'GCP_INSTANCE_NAME': self.get_instance_name()
        }
    
    def validate_config(self) -> bool:
        """Validate that all required GCP configuration is available"""
        try:
            self.get_all_config()
            return True
        except ValueError:
            return False
    
    def print_config_status(self):
        """Print current configuration status"""
        print("GCP Configuration Status:")
        print("=" * 30)
        
        project_id = os.getenv('GCP_PROJECT_ID')
        zone = os.getenv('GCP_ZONE')
        instance_name = os.getenv('GCP_INSTANCE_NAME')
        
        print(f"Project ID: {'✓ Set' if project_id else '✗ Missing'}")
        print(f"Zone: {'✓ Set' if zone else '✗ Missing'}")
        print(f"Instance: {'✓ Set' if instance_name else '✗ Missing'}")
        
        if not all([project_id, zone, instance_name]):
            print("\nTo set these values:")
            print("1. Copy env.example to .env.local")
            print("2. Edit .env.local with your GCP details")
            print("3. Or set environment variables directly")


def create_env_file_template():
    """Create a template .env file for the user"""
    template_content = """# TEP-GNSS Environment Configuration
# Copy this file to .env.local and fill in your values

# GCP Configuration (Required for GCP deployment)
GCP_PROJECT_ID=your-project-id-here
GCP_ZONE=us-central1-a
GCP_INSTANCE_NAME=your-instance-name-here

# Optional: Override analysis parameters
# TEP_WORKERS=16
# TEP_MEMORY_LIMIT_GB=12
"""
    
    env_file = Path(__file__).resolve().parents[2] / '.env.local'
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(template_content)
        print(f"Created template .env.local file at {env_file}")
        print("Please edit it with your GCP credentials")
    else:
        print(f".env.local already exists at {env_file}")


if __name__ == "__main__":
    # Test the configuration
    config = GCPConfig()
    config.print_config_status()
