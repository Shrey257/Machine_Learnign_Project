import os
import json
import time
import traceback

class ReportGenerator:
    """
    Class for generating HTML reports from player tracking data
    """
    def __init__(self, report_path):
        self.report_path = report_path
        self.frame_data = {}
        self.team_a_players = set()
        self.team_b_players = set()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        
        # Keep track of last processed frame for partial reports
        self.last_processed_frame = 0
        
    def update_frame_data(self, frame_number, player_teams, velocities, distances):
        """
        Update data for the current frame
        
        Args:
            frame_number: Current frame number
            player_teams: Dictionary mapping player IDs to teams
            velocities: Dictionary mapping player IDs to velocities
            distances: Dictionary mapping player IDs to distances
        """
        try:
            # Store a snapshot of the data for this frame
            self.frame_data[frame_number] = {
                'player_teams': player_teams.copy() if player_teams else {},
                'velocities': velocities.copy() if velocities else {},
                'distances': distances.copy() if distances else {}
            }
            
            # Update team membership sets
            for player_id, team in player_teams.items():
                if team == "A":
                    self.team_a_players.add(player_id)
                elif team == "B":
                    self.team_b_players.add(player_id)
                    
            # Update last processed frame
            self.last_processed_frame = max(self.last_processed_frame, frame_number)
        except Exception as e:
            print(f"Error updating frame data: {e}")
    
    def generate_report(self):
        """
        Generate an HTML report from the collected data
        """
        try:
            if not self.frame_data:
                print("No frame data available to generate report")
                self._generate_empty_report()
                return
                
            # Create player statistics
            player_stats = {}
            
            # Process frame data to calculate final statistics
            for frame_num, data in self.frame_data.items():
                try:
                    for player_id, team in data.get('player_teams', {}).items():
                        if player_id not in player_stats:
                            player_stats[player_id] = {
                                'id': player_id,
                                'team': team,
                                'max_speed': 0,
                                'avg_speed': 0,
                                'total_distance': 0,
                                'speed_samples': [],
                            }
                        
                        # Update speeds
                        if player_id in data.get('velocities', {}):
                            speed = data['velocities'][player_id]
                            # Handle nan and infinite values
                            if isinstance(speed, (int, float)) and speed >= 0 and speed < 100:  # Filter unrealistic speeds
                                player_stats[player_id]['speed_samples'].append(speed)
                                player_stats[player_id]['max_speed'] = max(player_stats[player_id]['max_speed'], speed)
                        
                        # Update distances
                        if player_id in data.get('distances', {}):
                            distance = data['distances'][player_id]
                            # Handle nan and infinite values
                            if isinstance(distance, (int, float)) and distance >= 0 and distance < 10000:  # Filter unrealistic distances
                                player_stats[player_id]['total_distance'] = distance
                except Exception as e:
                    print(f"Error processing frame {frame_num}: {e}")
                    continue
            
            # Calculate average speeds
            for player_id, stats in player_stats.items():
                try:
                    speed_samples = stats.get('speed_samples', [])
                    if speed_samples:
                        # Filter out outliers (speeds above 60 km/h are unlikely)
                        valid_samples = [s for s in speed_samples if 0 <= s <= 60]
                        if valid_samples:
                            stats['avg_speed'] = sum(valid_samples) / len(valid_samples)
                    # Remove the samples list from final output
                    if 'speed_samples' in stats:
                        del stats['speed_samples']
                except Exception as e:
                    print(f"Error calculating average speed for player {player_id}: {e}")
                    stats['avg_speed'] = 0
            
            # Convert to list for HTML template
            players_list = list(player_stats.values())
            
            # Calculate team statistics
            team_a_stats = self._calculate_team_stats([p for p in players_list if p.get('team') == 'A'])
            team_b_stats = self._calculate_team_stats([p for p in players_list if p.get('team') == 'B'])
            
            # Generate the HTML report
            self._generate_html_report(players_list, team_a_stats, team_b_stats)
            
            # Save a backup of the frame data to JSON
            try:
                backup_path = os.path.join(os.path.dirname(self.report_path), 'report_data_backup.json')
                with open(backup_path, 'w') as f:
                    # Only save essential data to keep file size small
                    simplified_data = {
                        'player_stats': player_stats,
                        'team_a_stats': team_a_stats,
                        'team_b_stats': team_b_stats,
                        'last_frame': self.last_processed_frame
                    }
                    json.dump(simplified_data, f)
            except Exception as e:
                print(f"Error saving backup data: {e}")
                
        except Exception as e:
            print(f"Error generating report: {e}")
            traceback.print_exc()
            # Try to generate a simple report with whatever data we have
            self._generate_emergency_report()
    
    def _calculate_team_stats(self, players):
        """Calculate team statistics from player data"""
        try:
            if not players:
                return {
                    'avg_max_speed': 0,
                    'avg_avg_speed': 0,
                    'total_distance': 0,
                    'player_count': 0
                }
            
            max_speeds = [p.get('max_speed', 0) for p in players]
            avg_speeds = [p.get('avg_speed', 0) for p in players]
            distances = [p.get('total_distance', 0) for p in players]
            
            # Filter out invalid values
            max_speeds = [s for s in max_speeds if isinstance(s, (int, float)) and 0 <= s <= 60]
            avg_speeds = [s for s in avg_speeds if isinstance(s, (int, float)) and 0 <= s <= 60]
            distances = [d for d in distances if isinstance(d, (int, float)) and 0 <= d <= 10000]
            
            return {
                'avg_max_speed': sum(max_speeds) / len(max_speeds) if max_speeds else 0,
                'avg_avg_speed': sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0,
                'total_distance': sum(distances),
                'player_count': len(players)
            }
        except Exception as e:
            print(f"Error calculating team stats: {e}")
            # Return default values if there's an error
            return {
                'avg_max_speed': 0,
                'avg_avg_speed': 0,
                'total_distance': 0,
                'player_count': len(players) if players else 0
            }
    
    def _generate_html_report(self, players, team_a_stats, team_b_stats):
        """Generate HTML report using the collected data"""
        try:
            # Filter out players with NaN or invalid values
            filtered_players = []
            for player in players:
                if (isinstance(player.get('max_speed'), (int, float)) and 
                    isinstance(player.get('avg_speed'), (int, float)) and 
                    isinstance(player.get('total_distance'), (int, float))):
                    filtered_players.append(player)
            
            # Sort players by team and then by total distance
            filtered_players.sort(key=lambda p: (p.get('team', 'Z') != 'A', -p.get('total_distance', 0)))
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Football Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr.team-a {{ background-color: #ffeeee; }}
                    tr.team-b {{ background-color: #eeeeff; }}
                    .team-summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                    .team-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; width: 48%; }}
                    .team-a-header {{ background-color: #ffcccc; padding: 10px; }}
                    .team-b-header {{ background-color: #ccccff; padding: 10px; }}
                </style>
            </head>
            <body>
                <h1>Football Match Analysis Report</h1>
                <p>Last processed frame: {self.last_processed_frame}</p>
                
                <div class="team-summary">
                    <div class="team-card">
                        <div class="team-a-header">
                            <h2>Team A Summary (White Jerseys)</h2>
                        </div>
                        <p>Players: {team_a_stats.get('player_count', 0)}</p>
                        <p>Average Max Speed: {team_a_stats.get('avg_max_speed', 0):.2f} km/h</p>
                        <p>Average Speed: {team_a_stats.get('avg_avg_speed', 0):.2f} km/h</p>
                        <p>Total Distance: {team_a_stats.get('total_distance', 0):.2f} m</p>
                    </div>
                    
                    <div class="team-card">
                        <div class="team-b-header">
                            <h2>Team B Summary (Green Jerseys)</h2>
                        </div>
                        <p>Players: {team_b_stats.get('player_count', 0)}</p>
                        <p>Average Max Speed: {team_b_stats.get('avg_max_speed', 0):.2f} km/h</p>
                        <p>Average Speed: {team_b_stats.get('avg_avg_speed', 0):.2f} km/h</p>
                        <p>Total Distance: {team_b_stats.get('total_distance', 0):.2f} m</p>
                    </div>
                </div>
                
                <h2>Player Statistics</h2>
                <table>
                    <tr>
                        <th>Player ID</th>
                        <th>Team</th>
                        <th>Max Speed (km/h)</th>
                        <th>Avg Speed (km/h)</th>
                        <th>Total Distance (m)</th>
                    </tr>
            """
            
            # Add player rows
            for player in filtered_players:
                team_class = "team-a" if player.get('team') == 'A' else "team-b"
                html_content += f"""
                    <tr class="{team_class}">
                        <td>{player.get('id', '')}</td>
                        <td>{player.get('team', '')}</td>
                        <td>{player.get('max_speed', 0):.2f}</td>
                        <td>{player.get('avg_speed', 0):.2f}</td>
                        <td>{player.get('total_distance', 0):.2f}</td>
                    </tr>
                """
            
            # Close the HTML
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            html_content += f"""
                </table>
                
                <h3>Notes:</h3>
                <ul>
                    <li>Team A players wear white jerseys</li>
                    <li>Team B players wear green jerseys</li>
                    <li>Speeds are in kilometers per hour</li>
                    <li>Distances are in meters</li>
                </ul>
                
                <p><em>Generated by Football Analysis System at {timestamp}</em></p>
            </body>
            </html>
            """
            
            # Write the HTML to file
            with open(self.report_path, 'w') as f:
                f.write(html_content)
            
            print(f"HTML report generated successfully at {self.report_path}")
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            traceback.print_exc()
            self._generate_emergency_report()
            
    def _generate_empty_report(self):
        """Generate an empty report when no data is available"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Football Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Football Match Analysis Report</h1>
            <p>No data available yet. The analysis is still in progress.</p>
            <p><em>Generated by Football Analysis System at {timestamp}</em></p>
        </body>
        </html>
        """
        
        # Write the HTML to file
        with open(self.report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Empty report generated at {self.report_path}")
            
    def _generate_emergency_report(self):
        """Generate a simplified report in case of errors"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Count players per team
            team_a_count = len(self.team_a_players)
            team_b_count = len(self.team_b_players)
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Football Analysis Report (Emergency)</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .error {{ color: #cc0000; }}
                </style>
            </head>
            <body>
                <h1>Football Match Analysis Report (Emergency Version)</h1>
                <p class="error">Note: This is an emergency report generated due to errors in processing.</p>
                <p>Last processed frame: {self.last_processed_frame}</p>
                
                <h2>Team Summary</h2>
                <p>Team A (White Jerseys): {team_a_count} players detected</p>
                <p>Team B (Green Jerseys): {team_b_count} players detected</p>
                
                <h3>Detected Players:</h3>
                <p>Team A IDs: {', '.join(map(str, sorted(self.team_a_players)))}</p>
                <p>Team B IDs: {', '.join(map(str, sorted(self.team_b_players)))}</p>
                
                <p><em>Emergency report generated at {timestamp}</em></p>
            </body>
            </html>
            """
            
            # Write the HTML to file
            with open(self.report_path, 'w') as f:
                f.write(html_content)
            
            print(f"Emergency HTML report generated at {self.report_path}")
        except Exception as e:
            print(f"Failed to generate emergency report: {e}")
            # Last resort - create minimal report
            with open(self.report_path, 'w') as f:
                f.write(f"<html><body><h1>Football Analysis Error Report</h1><p>Generated at {timestamp}</p></body></html>")


def generate_html_report(player_stats, output_path):
    """
    Legacy function for backward compatibility
    Generate an HTML report from player statistics
    
    Args:
        player_stats: Dictionary of player statistics
        output_path: Path to save the HTML report
    """
    # Create a new ReportGenerator
    report_gen = ReportGenerator(output_path)
    
    # Convert the player_stats to the format expected by ReportGenerator
    frame_data = {
        0: {
            'player_teams': {},
            'velocities': {},
            'distances': {}
        }
    }
    
    # Fill in the data
    for player_id, stats in player_stats.items():
        frame_data[0]['player_teams'][player_id] = stats.get('team', '')
        frame_data[0]['velocities'][player_id] = stats.get('max_speed', 0)
        frame_data[0]['distances'][player_id] = stats.get('total_distance', 0)
    
    # Update the report generator with the data
    report_gen.frame_data = frame_data
    
    # Generate the report
    report_gen.generate_report() 