import os
import pandas as pd
import numpy as np
import pyshark
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict
import statistics
import glob
import time
import csv
import warnings
warnings.filterwarnings('ignore')

class PcapAnalyzer:
    def __init__(self, pcap_dir='pcap', output_dir='output'):
        """
        Initialize PcapAnalyzer class
        
        Parameters:
            pcap_dir (str): Directory containing pcap files
            output_dir (str): Directory for output CSV files
        """
        self.pcap_dir = pcap_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Column names for basic features CSV
        self.basic_columns = [
            'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol',
            'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
            'label'
        ]
        
        # Column names for advanced features CSV (with window features)
        self.advanced_columns = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']
        
        # Add features for each window
        window_features = [
            'Packets_Count', 'Bytes_Count', 'IAT_Mean', 'IAT_Std',
            'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Bwd_IAT_Mean', 'Bwd_IAT_Std'
        ]
        
        for i in range(10):  # 10 windows
            for feature in window_features:
                self.advanced_columns.append(f'Window_{i+1}_{feature}')
                
        self.advanced_columns.append('label')
    
    def extract_label_from_filename(self, filename):
        """Extract label from filename (content before first "_")"""
        base_name = os.path.basename(filename)
        return base_name.split('_')[0]
    
    def process_pcap_files(self):
        """Process all pcap files and generate CSV files"""
        # Get all pcap files
        pcap_files = glob.glob(os.path.join(self.pcap_dir, '*.pcap'))
        
        if not pcap_files:
            print(f"No pcap files found in {self.pcap_dir} directory")
            return
        
        # Create DataFrames for basic and advanced features
        basic_df = pd.DataFrame(columns=self.basic_columns)
        advanced_df = pd.DataFrame(columns=self.advanced_columns)
        
        # Process each pcap file
        for pcap_file in pcap_files:
            print(f"Processing: {pcap_file}")
            label = self.extract_label_from_filename(pcap_file)
            
            try:
                # Process pcap file to extract flows and features
                basic_features, advanced_features = self.extract_features_from_pcap(pcap_file, label)
                
                if basic_features and advanced_features:
                    # Append features to DataFrames
                    basic_df = pd.concat([basic_df, pd.DataFrame(basic_features, columns=self.basic_columns)])
                    advanced_df = pd.concat([advanced_df, pd.DataFrame(advanced_features, columns=self.advanced_columns)])
                    
            except Exception as e:
                print(f"Error processing {pcap_file}: {e}")
        
        # Save DataFrames to CSV files
        if not basic_df.empty and not advanced_df.empty:
            basic_csv_path = os.path.join(self.output_dir, 'basic_features.csv')
            advanced_csv_path = os.path.join(self.output_dir, 'advanced_features.csv')
            
            basic_df.to_csv(basic_csv_path, index=False)
            advanced_df.to_csv(advanced_csv_path, index=False)
            
            print(f"Basic features saved to: {basic_csv_path}")
            print(f"Advanced features saved to: {advanced_csv_path}")
            
            return basic_csv_path, advanced_csv_path
        else:
            print("No features extracted or all processing failed")
            return None, None
    
    def extract_features_from_pcap(self, pcap_file, label):
        """
        Extract basic and advanced features from a pcap file
        
        Parameters:
            pcap_file (str): Path to pcap file
            label (str): Traffic class label
            
        Returns:
            tuple: (basic_features, advanced_features)
        """
        try:
            # Read pcap file
            packets = rdpcap(pcap_file)
            
            if not packets:
                print(f"No packets found in {pcap_file}")
                return [], []
            
            # Group packets into flows
            flows = self.group_packets_into_flows(packets)
            
            # Extract features for each flow
            basic_features = []
            advanced_features = []
            
            for flow_key, flow_packets in flows.items():
                # Split flow into subflows (10-second windows)
                subflows = self.split_flow_into_subflows(flow_packets)
                
                for subflow in subflows:
                    if len(subflow) < 2:  # Need at least 2 packets for valid features
                        continue
                    
                    # Extract basic features
                    basic_feature = self.calculate_basic_features(subflow, flow_key, label)
                    if basic_feature:
                        basic_features.append(basic_feature)
                    
                    # Extract advanced features (temporal)
                    advanced_feature = self.calculate_advanced_features(subflow, flow_key, label)
                    if advanced_feature:
                        advanced_features.append(advanced_feature)
            
            return basic_features, advanced_features
            
        except Exception as e:
            print(f"Error extracting features from {pcap_file}: {e}")
            return [], []
    
    def group_packets_into_flows(self, packets):
        """
        Group packets into flows based on 5-tuple
        
        Parameters:
            packets: List of packets from pcap file
            
        Returns:
            dict: Dictionary of flows, keyed by flow 5-tuple
        """
        flows = defaultdict(list)
        
        for packet in packets:
            if IP in packet:
                if TCP in packet:
                    proto = 'TCP'
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                elif UDP in packet:
                    proto = 'UDP'
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                else:
                    continue  # Skip non-TCP/UDP packets
                
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                # Create bidirectional flow key (smaller IP first for consistency)
                if src_ip < dst_ip:
                    flow_key = (src_ip, src_port, dst_ip, dst_port, proto)
                else:
                    flow_key = (dst_ip, dst_port, src_ip, src_port, proto)
                
                flows[flow_key].append(packet)
        
        return flows
    
    def split_flow_into_subflows(self, flow_packets, window_size=10):
        """
        Split flow into subflows of specified window size
        
        Parameters:
            flow_packets: List of packets in the flow
            window_size (int): Window size in seconds
            
        Returns:
            list: List of subflows (lists of packets)
        """
        if not flow_packets:
            return []
        
        # Sort packets by time
        flow_packets.sort(key=lambda p: p.time)
        
        subflows = []
        current_subflow = []
        start_time = flow_packets[0].time
        
        for packet in flow_packets:
            # If packet is within current window, add to current subflow
            if packet.time - start_time <= window_size:
                current_subflow.append(packet)
            else:
                # Start a new subflow
                if current_subflow:
                    subflows.append(current_subflow)
                current_subflow = [packet]
                start_time = packet.time
        
        # Add the last subflow
        if current_subflow:
            subflows.append(current_subflow)
        
        return subflows
    
    def calculate_basic_features(self, packets, flow_key, label):
        """
        Calculate basic flow features
        
        Parameters:
            packets: List of packets in the flow
            flow_key: Flow identifier tuple
            label: Traffic class label
            
        Returns:
            list: List of feature values
        """
        try:
            src_ip, src_port, dst_ip, dst_port, proto = flow_key
            
            # Flow duration
            start_time = packets[0].time
            end_time = packets[-1].time
            duration = end_time - start_time
            
            # Skip very short flows
            if duration < 0.001:
                return None
            
            # Calculate flow bytes and packets per second
            total_bytes = sum(len(p) for p in packets)
            total_packets = len(packets)
            
            bytes_per_sec = total_bytes / duration
            packets_per_sec = total_packets / duration
            
            # Calculate packet inter-arrival times (IAT)
            packet_times = [p.time for p in packets]
            iats = [packet_times[i+1] - packet_times[i] for i in range(len(packet_times)-1)]
            
            # Calculate IAT statistics
            if iats:
                iat_mean = statistics.mean(iats)
                iat_std = statistics.stdev(iats) if len(iats) > 1 else 0
                iat_max = max(iats)
                iat_min = min(iats)
            else:
                iat_mean = iat_std = iat_max = iat_min = 0
            
            # Separate forward and backward packets
            fwd_packets = [p for p in packets if p[IP].src == src_ip]
            bwd_packets = [p for p in packets if p[IP].src == dst_ip]
            
            # Calculate forward IAT statistics
            fwd_times = [p.time for p in fwd_packets]
            fwd_iats = [fwd_times[i+1] - fwd_times[i] for i in range(len(fwd_times)-1)]
            
            if fwd_iats:
                fwd_iat_mean = statistics.mean(fwd_iats)
                fwd_iat_std = statistics.stdev(fwd_iats) if len(fwd_iats) > 1 else 0
                fwd_iat_max = max(fwd_iats)
                fwd_iat_min = min(fwd_iats)
            else:
                fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0
            
            # Calculate backward IAT statistics
            bwd_times = [p.time for p in bwd_packets]
            bwd_iats = [bwd_times[i+1] - bwd_times[i] for i in range(len(bwd_times)-1)]
            
            if bwd_iats:
                bwd_iat_mean = statistics.mean(bwd_iats)
                bwd_iat_std = statistics.stdev(bwd_iats) if len(bwd_iats) > 1 else 0
                bwd_iat_max = max(bwd_iats)
                bwd_iat_min = min(bwd_iats)
            else:
                bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0
            
            # Calculate active and idle times (simplified)
            # Using a threshold of 1 second for idle time
            idle_threshold = 1.0
            
            active_times = []
            idle_times = []
            
            current_active_start = packet_times[0]
            
            for i in range(len(iats)):
                if iats[i] > idle_threshold:
                    # End of active period
                    active_time = packet_times[i] - current_active_start
                    active_times.append(active_time)
                    idle_times.append(iats[i])
                    current_active_start = packet_times[i+1]
            
            # Add the last active period
            active_times.append(packet_times[-1] - current_active_start)
            
            # Calculate active/idle statistics
            if active_times:
                active_mean = statistics.mean(active_times)
                active_std = statistics.stdev(active_times) if len(active_times) > 1 else 0
                active_max = max(active_times)
                active_min = min(active_times)
            else:
                active_mean = active_std = active_max = active_min = 0
                
            if idle_times:
                idle_mean = statistics.mean(idle_times)
                idle_std = statistics.stdev(idle_times) if len(idle_times) > 1 else 0
                idle_max = max(idle_times)
                idle_min = min(idle_times)
            else:
                idle_mean = idle_std = idle_max = idle_min = 0
            
            # Combine all features
            features = [
                src_ip, src_port, dst_ip, dst_port, proto,
                duration, bytes_per_sec, packets_per_sec,
                iat_mean, iat_std, iat_max, iat_min,
                fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min,
                bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min,
                active_mean, active_std, active_max, active_min,
                idle_mean, idle_std, idle_max, idle_min,
                label
            ]
            
            return features
            
        except Exception as e:
            print(f"Error calculating basic features: {e}")
            return None
    
    def calculate_advanced_features(self, packets, flow_key, label):
        """
        Calculate advanced temporal features by dividing the flow into 10 windows
        
        Parameters:
            packets: List of packets in the flow
            flow_key: Flow identifier tuple
            label: Traffic class label
            
        Returns:
            list: List of feature values
        """
        try:
            src_ip, src_port, dst_ip, dst_port, proto = flow_key
            
            # Sort packets by time
            packets.sort(key=lambda p: p.time)
            
            # Calculate total duration
            start_time = packets[0].time
            end_time = packets[-1].time
            total_duration = end_time - start_time
            
            # Skip very short flows
            if total_duration < 0.01:
                return None
            
            # Divide packets into 10 equal-sized windows
            window_size = total_duration / 10
            windows = [[] for _ in range(10)]
            
            for packet in packets:
                # Determine which window this packet belongs to
                time_offset = packet.time - start_time
                window_idx = min(9, int(time_offset / window_size))
                windows[window_idx].append(packet)
            
            # Initialize features with flow identification
            features = [src_ip, src_port, dst_ip, dst_port, proto]
            
            # Calculate features for each window
            for window in windows:
                if not window:
                    # If window is empty, add zero values
                    features.extend([0, 0, 0, 0, 0, 0, 0, 0])
                    continue
                
                # Calculate packet count and byte count
                packets_count = len(window)
                bytes_count = sum(len(p) for p in window)
                
                # Calculate IAT
                packet_times = [p.time for p in window]
                iats = [packet_times[i+1] - packet_times[i] for i in range(len(packet_times)-1)]
                
                iat_mean = statistics.mean(iats) if iats else 0
                iat_std = statistics.stdev(iats) if len(iats) > 1 else 0
                
                # Separate forward and backward packets
                fwd_packets = [p for p in window if p[IP].src == src_ip]
                bwd_packets = [p for p in window if p[IP].src == dst_ip]
                
                # Calculate forward IAT
                fwd_times = [p.time for p in fwd_packets]
                fwd_iats = [fwd_times[i+1] - fwd_times[i] for i in range(len(fwd_times)-1)]
                
                fwd_iat_mean = statistics.mean(fwd_iats) if fwd_iats else 0
                fwd_iat_std = statistics.stdev(fwd_iats) if len(fwd_iats) > 1 else 0
                
                # Calculate backward IAT
                bwd_times = [p.time for p in bwd_packets]
                bwd_iats = [bwd_times[i+1] - bwd_times[i] for i in range(len(bwd_times)-1)]
                
                bwd_iat_mean = statistics.mean(bwd_iats) if bwd_iats else 0
                bwd_iat_std = statistics.stdev(bwd_iats) if len(bwd_iats) > 1 else 0
                
                # Add window features
                features.extend([
                    packets_count, bytes_count, iat_mean, iat_std,
                    fwd_iat_mean, fwd_iat_std, bwd_iat_mean, bwd_iat_std
                ])
            
            # Add label
            features.append(label)
            
            return features
            
        except Exception as e:
            print(f"Error calculating advanced features: {e}")
            return None

def main():
    """Main function"""
    start_time = time.time()
    
    # Create PcapAnalyzer instance
    analyzer = PcapAnalyzer(pcap_dir='pcap', output_dir='output')
    
    # Process pcap files
    basic_csv, advanced_csv = analyzer.process_pcap_files()
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    if basic_csv and advanced_csv:
        print(f"Basic features CSV: {basic_csv}")
        print(f"Advanced features CSV: {advanced_csv}")

if __name__ == "__main__":
    main()
