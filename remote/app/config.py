import os
import json

# Define the configuration file path
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_config.json")

class SiteConfig:
    def __init__(self, name="", central_url="ws://127.0.0.1:8000", http_url="http://127.0.0.1:8000", 
                 site_id="my_site_id", token="my_jwt_token"):
        self.name = name
        self.CENTRAL_URL = central_url
        self.HTTP_URL = http_url
        self.SITE_ID = site_id
        self.TOKEN = token

class Settings:
    def __init__(self):
        self.sites = []
        
    # Remove current_site property - always use explicit site selection
    # Keep backwards compatibility properties that reference the first site
    @property
    def CENTRAL_URL(self):
        return self.sites[0].CENTRAL_URL if self.sites else "ws://127.0.0.1:8000"
    
    @CENTRAL_URL.setter
    def CENTRAL_URL(self, value):
        if self.sites:
            self.sites[0].CENTRAL_URL = value
    
    @property
    def HTTP_URL(self):
        return self.sites[0].HTTP_URL if self.sites else "http://127.0.0.1:8000"
    
    @HTTP_URL.setter
    def HTTP_URL(self, value):
        if self.sites:
            self.sites[0].HTTP_URL = value
    
    @property
    def SITE_ID(self):
        return self.sites[0].SITE_ID if self.sites else "my_site_id"
    
    @SITE_ID.setter
    def SITE_ID(self, value):
        if self.sites:
            self.sites[0].SITE_ID = value
    
    @property
    def TOKEN(self):
        return self.sites[0].TOKEN if self.sites else "my_jwt_token"
    
    @TOKEN.setter
    def TOKEN(self, value):
        if self.sites:
            self.sites[0].TOKEN = value

def save_settings(settings):
    """Save settings to a local JSON file for persistence"""
    try:
        sites_data = []
        for site in settings.sites:
            sites_data.append({
                "name": site.name,
                "CENTRAL_URL": site.CENTRAL_URL,
                "HTTP_URL": site.HTTP_URL,
                "SITE_ID": site.SITE_ID,
                "TOKEN": site.TOKEN
            })
        
        data = {
            "sites": sites_data
        }
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Settings saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def load_settings():
    """Load settings from local JSON file if it exists"""
    settings = Settings()
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                
                if "sites" in data:
                    # New format with multiple sites
                    for site_data in data.get("sites", []):
                        site = SiteConfig(
                            name=site_data.get("name", ""),
                            central_url=site_data.get("CENTRAL_URL", "ws://127.0.0.1:8000"),
                            http_url=site_data.get("HTTP_URL", "http://127.0.0.1:8000"),
                            site_id=site_data.get("SITE_ID", "my_site_id"),
                            token=site_data.get("TOKEN", "my_jwt_token")
                        )
                        settings.sites.append(site)
                else:
                    # Old format with a single site
                    site = SiteConfig(
                        name="Site 1",
                        central_url=data.get("CENTRAL_URL", "ws://127.0.0.1:8000"),
                        http_url=data.get("HTTP_URL", "http://127.0.0.1:8000"),
                        site_id=data.get("SITE_ID", "my_site_id"),
                        token=data.get("TOKEN", "my_jwt_token")
                    )
                    settings.sites.append(site)
                
                print(f"Settings loaded from {CONFIG_FILE}")
    except Exception as e:
        print(f"Error loading settings: {e}")
        # Create a default site if no configuration is loaded
        if not settings.sites:
            site = SiteConfig(name="Site 1")
            settings.sites.append(site)
    
    return settings

# Load or create settings
settings = load_settings()
